import pickle

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time

from dgl import load_graphs
from rdkit import Chem
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.distributions import DistributionProperty
from models.LeftNet import LEFTNet
from models.optimizer import NAdamW
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete, LEFTLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, SumExceptBatchMSE, SumExceptBatchMAE, SumExceptBatchWasserstein
import utils
from process_data import process_mol
from utils import center_pos, normalize_prop, prepare_context, padding, ex_batch, zero_module
from line_profiler import profile
from pathlib import Path
from analysis.rdkit_functions import build_molecule, build_molecule_with_partial_charges, mol2smiles
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_batch

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, property_norms, property_norms_val, regressor_train=False, guidance_model=None, switch_epoch=30):
        super().__init__()

        self.prop_dist = None
        # add for test
        self.guidance_model = guidance_model
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(cfg.general, False)

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.atom_type_to_atomic_number = cfg.dataset.atom_type_to_atomic_number

        self.property_norms = property_norms
        self.property_norms_val = property_norms_val

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.model1_train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.model2_loss = LEFTLossDiscrete(self.atom_type_to_atomic_number)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_pos_mse = SumExceptBatchMSE()
        self.val_atomic = SumExceptBatchKL()
        self.val_pos_wasser = SumExceptBatchWasserstein()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_pos_mse = SumExceptBatchMSE()
        self.test_atomic = SumExceptBatchKL()
        self.test_pos_wasser = SumExceptBatchWasserstein()
        self.test_cond_loss = MeanAbsoluteError()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model1 = GraphTransformer(n_layers=cfg.model.n_layers,
                                       input_dims=input_dims,
                                       hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                       hidden_dims=cfg.model.hidden_dims,
                                       output_dims=output_dims,
                                       act_fn_in=nn.ReLU(),
                                       act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=self.T)

        # self.model2 = DualEdgeEGNN(cfg=self.cfg)
        self.cfg.general.regressor_train=regressor_train
        if self.cfg.general.regressor_train:
            self.model2 = LEFTNet(cfg=self.cfg, property_norms=self.property_norms)
        else:
            self.model2 = LEFTNet(cfg=self.cfg)

        self.optimizer = cfg.train.optimizer
        self.switch_epoch=switch_epoch

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")

            y_marginals = torch.tensor([0.4,0.1,0.4,0.1])
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_marginals=y_marginals)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,  
                                                y=y_marginals)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics', 'dataset_info'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        # self.accumulation_steps=4
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.num_valid_molecules = 0
        self.num_total = 0
        # self.automatic_optimization = False

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask= utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        ### local edge added
        # if self.cfg.general.regressor:
        context = prepare_context(self.cfg.model.context, data, self.property_norms)
        # else:
        # context=None

        pred, net_out, atomic_t, pos_t = self.forward(noisy_data, extra_data, node_mask,
                                     data.pos, data.batch, context)

        if self.current_epoch % (2 * self.switch_epoch) < self.switch_epoch:
            loss = self.model1_train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, node_mask=node_mask,
                                           pred_y=pred.y,
                                           true_X=X, true_E=E, true_y=data.y,
                                           log=i % self.log_every_steps == 0)
        else:
            loss1 = self.model1_train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, node_mask=node_mask,
                                           pred_y=pred.y,
                                           true_X=X, true_E=E, true_y=data.y,
                                           log=i % self.log_every_steps == 0)
            loss2 = self.model2_loss(net_out, X, data.pos, node_mask, data.batch,
                                     log=i % self.log_every_steps == 0)
            loss = loss1 + loss2
        
        # loss1 = self.model1_train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, node_mask=node_mask, pred_y=pred.y,
        #                                     true_X=X, true_E=E, true_y=y, 
        #                                     log=i % self.log_every_steps == 0) 
        # loss2 = self.model2_loss(net_out, pos, X, node_mask, log=i % self.log_every_steps ==0)
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)


        return {'loss': loss}

    
    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_model(model):
        for param in model.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            # optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.cfg.train.lr, amsgrad=True,
            #                             weight_decay=self.cfg.train.weight_decay)
            # optimizer2 = torch.optim.AdamW(self.model2.parameters(), lr=self.cfg.train.lr, amsgrad=True,
            #                             weight_decay=self.cfg.train.weight_decay)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                        weight_decay=self.cfg.train.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                        weight_decay=self.cfg.train.weight_decay)

        else:
            optimizer = NAdamW(self.parameters(), lr=self.cfg.train.lr, 
                                weight_decay=self.cfg.train.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer #, 'lr_scheduler': {"scheduler":lr_scheduler, 'interval': 'epoch', "monitor": "loss"}}
    
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        if self.cfg.general.regressor:
            self.prop_dist = DistributionProperty(self.trainer.datamodule.train_dataloader(),
                                                {k: v for k,v in self.dataset_info.prop2idx.items() if k == self.cfg.model.context[0]})
            self.prop_dist.set_normalizer(self.property_norms)
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_test_start(self) -> None:
        if self.cfg.general.regressor:
            self.prop_dist = DistributionProperty(self.trainer.datamodule.test_dataloader(),
                                                {k: v for k,v in self.dataset_info.prop2idx.items() if k in self.args.model.context[0]})
            self.prop_dist.set_normalizer(self.property_norms)
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 1:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.model1_train_loss.reset()
        self.model2_loss.reset()
        self.train_metrics.reset()
        if self.current_epoch % (2 * self.switch_epoch) < self.switch_epoch:
            self.freeze_model(self.model2)  # 冻结 model2
            self.unfreeze_model(self.model1)  # 解冻 model1
        else:
            self.freeze_model(self.model1)  # 冻结 model1
            self.unfreeze_model(self.model2)  # 解冻 model2

    def on_train_epoch_end(self) -> None:
        trans_to_log = self.model1_train_loss.log_epoch_metrics()
        pos_to_log = self.model2_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: -- "
                   f"X_CE: {trans_to_log['train_epoch/X_CE'] :.3f}"
                   f" -- E_CE: {trans_to_log['train_epoch/E_CE'] :.3f} --"
                   f" y_CE: {trans_to_log['train_epoch/y_CE'] :.3f} --"
                   f" pos_loss: {pos_to_log['train_epoch/pos_loss'] :.3f} --"
                   f" atomic_loss: {pos_to_log['train_epoch/atomic_loss'] :.3f} --"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_pos_mse.reset()
        self.val_atomic.reset()
        self.val_pos_wasser.reset()
        self.sampling_metrics.reset()
        if self.current_epoch % (2 * self.switch_epoch) < self.switch_epoch:
            self.freeze_model(self.model2)  # 冻结 model2
            self.unfreeze_model(self.model1)  # 解冻 model1
        else:
            self.freeze_model(self.model1)  # 冻结 model1
            self.unfreeze_model(self.model2)  # 解冻 model2

    def validation_step(self, data, i):
        dense_data, node_mask= utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)

        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        ### lcoal edge added
        # if self.cfg.general.regressor:
        context = prepare_context(self.cfg.model.context, data, self.property_norms_val)
        # else:
        # context=None
        pred, net_out, atomic_t, pos_t = self.forward(noisy_data, extra_data, node_mask,
                                     data.pos, data.batch, context)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask,
                                    net_out, data.pos, i, test=False)

        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_pos_mse.compute()* self.T, self.val_atomic.compute() * self.T,
                   self.val_pos_wasser.compute() * self.T]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4], 
                       'val/pos_mse': metrics[5],
                       'val/atomic_kl': metrics[6],
                       'val/pos_wasser': metrics[7],}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL: {metrics[0] :.2f} -- Val Atom type KL: {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f} -- ", #Val y KL: {metrics[3] :.2f} --",
                   f"Val X_logp: {metrics[3] :.2f} -- Val E_logp: {metrics[4] :.2f} --",
                   f"Val pos MSE: {metrics[5] : 2f} -- val atomic KL: {metrics[6] : 2f} -- ",
                   f"Val pos wasserstein distance: {metrics[7] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_pos_mse.reset()
        self.test_atomic.reset()
        self.test_pos_wasser.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, data, i):
        # dense_data, node_mask= utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        #
        # dense_data = dense_data.mask(node_mask)
        # noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        # extra_data = self.compute_extra_data(noisy_data)
        #
        # ### local edge added
        # # if self.cfg.general.regressor:
        # context = prepare_context(self.cfg.model.context, data, self.property_norms_val)
        # # else:
        # #     context=None
        # pred, net_out, atomic_t, pos_t = self.forward(noisy_data, extra_data, node_mask,
        #                              data.pos, data.batch, context)
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask,
        #                             net_out, data.pos, i, test=True)

        # guidance

        target_properties = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        ident = 0
        samples = self.sample_batch(batch_id=ident, batch_size=256, num_nodes=None,
                                    save_final=10,
                                    input_properties=target_properties)

        mae = self.save_cond_samples(samples, target_properties, file_path=f'cond_smiles{i}.pkl')
        # save conditional generated samples


        return {'loss': mae}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        final_mae = self.test_cond_loss.compute()
        final_validity = self.num_valid_molecules / self.num_total
        print("Final MAE", final_mae)
        print("Final validity", final_validity * 100)

        wandb.run.summary['final_MAE'] = final_mae
        wandb.run.summary['final_validity'] = final_validity
        wandb.log({'final mae': final_mae,
                   'final validity': final_validity})


        self.print("Saving the generated graphs")

    def save_cond_samples(self, samples, target, file_path):
        cond_results = {'smiles': [], 'input_targets': target, 'atom_type': [], 'edge_type': [], 'position': []}
        mols =[]

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            mol_2, mol_3 = build_molecule(sample[0], sample[1], sample[2], self.dataset_info.atom_decoder,
                                          self.dataset_info)
            smiles_2 = mol2smiles(mol_2)
            smiles_3 = mol2smiles(mol_3)
            if smiles_2 is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol_2, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol_2, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    cond_results['smiles'].append(smiles)
                    mols.append(largest_mol)
                    cond_results['atom_type'].append(sample[0])
                    cond_results['edge_type'].append(sample[1])
                    cond_results['position'].append(sample[2])
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
            elif smiles_3 is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol_3, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol_3, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    cond_results['smiles'].append(smiles)
                    mols.append(largest_mol)
                    cond_results['atom_type'].append(sample[0])
                    cond_results['edge_type'].append(sample[1])
                    cond_results['position'].append(sample[2])
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
            else:
                continue


        # save samples
        with open(file_path, 'wb') as f:
            pickle.dump(cond_results, f)
        pp_graph_list, _ = load_graphs(f"../../../../data/{self.cfg.model.context[0].split('_')[0]}_pdb/"
                                       f"{self.cfg.model.context[0].split('_')[0]}_phar_graphs.bin")
        for pp_graph in pp_graph_list:
            pp_graph.ndata['h'] = \
                torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
        loaded_reg = joblib.load('../../../../data/stacking_regressor_model_1.pkl')
        pred_results = [torch.tensor(process_mol((mol, pp_graph_list, loaded_reg))) for mol in mols]
        properties = torch.stack(pred_results, dim=0)
        mae = self.test_cond_loss(properties,
                            target[:properties.shape[0], :].cpu())
        self.num_valid_molecules += len(mols)
        self.num_total += len(samples)
        return mae

    
    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0,_,_,_ = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        time_step = torch.randint(lowest_t, self.T, size=(X.size(0) // 2 +1,), device=X.device) # (bs, 1)
        time_step = torch.cat(
            [time_step, self.T - time_step - 1], dim=0)[:X.size(0)]
        t_int = time_step.unsqueeze(1).long()
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=X.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=X.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'time_step': time_step,
                      'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 
                      'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, net_out, pos, i, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        if self.current_epoch % (2 * self.switch_epoch) < self.switch_epoch:
            N = node_mask.sum(1).long()
            log_pN = self.node_dist.log_prob(N)

            # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
            kl_prior = self.kl_prior(X, E, node_mask)

            # 3. Diffusion loss
            loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

            # 4. Reconstruction loss
            # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
            prob0 = self.reconstruction_logp(t, X, E, node_mask)

            loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        
            nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
            assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'
            if wandb.run:
                wandb.log({"kl prior": kl_prior.mean(),
                        "Estimator loss terms": loss_all_t.mean(),
                        "log_pn": log_pN.mean(),
                        "loss_term_0": loss_term_0}, commit=False)
        else:
            N = node_mask.sum(1).long()
            log_pN = self.node_dist.log_prob(N)

            # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
            kl_prior = self.kl_prior(X, E, node_mask)

            # 3. Diffusion loss
            loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

            # 4. Reconstruction loss
            # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
            prob0 = self.reconstruction_logp(t, X, E, node_mask)

            loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

            node_gt, pos_gt = net_out
            loss_pos = (self.test_pos_mse if test else self.val_pos_mse)(pos_gt[node_mask], pos)
            loss_atomic = (self.test_atomic if test else self.val_atomic)(X, node_gt)
            loss_wasser = self.val_pos_wasser(pos_gt[node_mask], pos)
            nlls = 0.01 * loss_pos + loss_atomic+loss_wasser - log_pN + kl_prior + loss_all_t - loss_term_0
            # Combine terms

            assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'
            if wandb.run:
                wandb.log({"kl prior": kl_prior.mean(),
                           "Estimator loss terms": loss_all_t.mean(),
                           "log_pn": log_pN.mean(),
                           "loss_term_0": loss_term_0,
                           "loss_pos": loss_pos,
                           "loss_atomic": loss_atomic,
                           "loss_wasser": loss_wasser}, commit=False)

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch
        if wandb.run:
            wandb.log({'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        
        return nll

    def forward(self, noisy_data, extra_data, node_mask, 
                pos=None, batch=None, context=None):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        output1 = self.model1(X, E, y, node_mask)
        if pos==None:
            output2, atomic_t, pos_t=None, None, None
        else:
            atomic_t, pos_t = self.left_process(pos, batch, output1.X, noisy_data['beta_t'], node_mask)
            output2 = self.model2(z=atomic_t,
                                  node_mask=node_mask,
                                  pos_perturbed=pos_t,
                                  batch=batch,
                                  context=context,
                                  time_step=noisy_data['time_step'])
        return output1, output2, atomic_t, pos_t

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, 
                     save_final: int, num_nodes=None, input_properties=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
            if self.prop_dist is not None:
                key_list = list(list(self.prop_dist.distributions.values())[0].keys())
                # #qm9
                mask = torch.tensor([node in key_list for node in n_nodes], dtype=torch.bool, device=n_nodes.device)
                n_nodes = torch.where(mask, n_nodes, torch.tensor(torch.max(n_nodes).item()))
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
    
        assert (E == torch.transpose(E, 1, 2)).all()
        batch = ex_batch(node_mask)
        pos_init = torch.randn(X.size(0), n_max, 3, device=X.device)
        pos_init = (pos_init * node_mask.unsqueeze(-1))[node_mask]
        pos = center_pos(pos_init, batch)

        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch_(n_nodes).to(self.device)
            context = context.index_select(0, batch)
        else:
            context=None
        # context[:, 0] = (context[:, 0] - context[:, 0].min()) / (context[:, 0].max() - context[:, 0].min())  #qm9
        
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)  #j
            t_array = s_array + 1 #i
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, pos = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, pos, batch, context, input_properties)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        pos, _ = to_dense_batch(pos, batch)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            positions = pos[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types, positions])

        # Visualize
        if self.visualization_tools is not None:
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, pos, batch, context, input_properties):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        time_step = (t*self.T).squeeze(1).long()

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask, 'beta_t': beta_t, 'time_step': time_step}
        extra_data = self.compute_extra_data(noisy_data)
        pred, net_out, _, _ = self.forward(noisy_data, extra_data, node_mask, pos, batch, context)
        _, pred_pos = net_out

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # # Guidance
        if self.guidance_model is not None:
            lamb = self.args.guidance.lambda_guidance

            grad_x, grad_e = self.cond_fn(noisy_data, node_mask, pos, batch, context, input_properties)

            p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
            p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

            prob_X_unnormalized = p_eta_x * prob_X
            prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
            prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

            prob_E_unnormalized = p_eta_e * prob_E
            prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
            prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        noise = center_pos(torch.randn_like(pos), batch)  #batch, 3
        pos_, _ = to_dense_batch(pos, batch)
        e = -(pred_pos - pos_)
        alphas = self.noise_schedule.alphas.to(pos.device)
        alpha_t = alphas[time_step]
        alpha_s = alphas[(s*self.T).squeeze(1).long()]
        betas_t = 1 - alpha_t / alpha_s  #bs
        beta_t_expanded = betas_t.index_select(0, batch) #batch,

        mean = (pos - (betas_t.view(-1,1,1) * e)[node_mask]) / (1 - beta_t_expanded).clamp(min=1e-9).sqrt().unsqueeze(-1)  #(G, 3)
        mask = 1 - (time_step[0] == 0).float()
        logvar = beta_t_expanded.log().unsqueeze(-1)  #(G, 1)
        pos_next = mean + mask * torch.exp(
            0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ  (G, 3)

        return out_one_hot.mask(node_mask).type_as(y_t), pos_next


    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
    
    def left_process(self, pos, batch, X, betas, node_mask):
        pos, _ = to_dense_batch(x=pos, batch=batch)     
        pos = pos.float() * node_mask.unsqueeze(-1)
        alphas = (1. - torch.clamp(betas, min=0, max=0.999)).cumprod(dim=0)
        # a = alphas.index_select(0, time_step)  # (G, )

        a_pos = alphas.index_select(0, batch)  # (N, 1)

        """
        Independently
        - Perterb pos
        """
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos[node_mask] + center_pos(pos_noise[node_mask], batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()  #batch, 3
        """
        Perterb atom
        """
        X_exist = torch.argmax(X, dim=-1)
        atomic_t = X_exist.cpu().apply_(lambda x: self.atom_type_to_atomic_number.get(x, x)).to(X.device)
        atomic_t = atomic_t * node_mask

        # atom_noise = torch.zeros(size=atomic_t.size(), device=atomic_t.device)
        # atom_noise.normal_()  #bs, n
        # atom_type = torch.cat([atomic_t[:, :-1] / 4, atomic_t[:, -1:] / 10], dim=1)
        # atom_perturbed = a_pos.sqrt() * atom_type[node_mask].unsqueeze(-1) + (1.0 - a_pos).sqrt() * atom_noise[node_mask].unsqueeze(-1) #batch,1
        return atomic_t[node_mask], pos_perturbed

    def cond_fn(self, noisy_data, node_mask,  pos, batch, context, target=None):
        # self.guidance_model.eval()
        loss = nn.MSELoss()

        t = noisy_data['t']

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['t']
        atomic_t, pos_t = self.left_process(pos, batch, X, noisy_data['beta_t'], node_mask)
        with torch.enable_grad():
            x_in = X.float().detach().requires_grad_(True)
            e_in = E.float().detach().requires_grad_(True)

            pred = self.guidance_model.model1(x_in, e_in, y, node_mask)
            net_out = self.guidance_model.model2(z=atomic_t,
                                                  node_mask=node_mask,
                                                  pos_perturbed=pos_t,
                                                  batch=batch,
                                                  context=context,
                                                  time_step=noisy_data['time_step'])

            # normalize target
            target = target.type_as(x_in)

            mse = loss(pred.y, target) + loss(net_out, target[:, 0].unsqueeze(1))

            t_int = int(t[0].item() * 500)
            if t_int % 10 == 0:
                print(f'Regressor MSE at step {t_int}: {mse.item()}')
            wandb.log({'Guidance MSE': mse})

            # calculate gradient of mse with respect to x and e
            grad_x = torch.autograd.grad(mse, x_in, retain_graph=True)[0]
            grad_e = torch.autograd.grad(mse, e_in)[0]

            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            diag_mask = torch.eye(n)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            mask_grad_e = 1 / 2 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))
            return mask_grad_x, mask_grad_e
    