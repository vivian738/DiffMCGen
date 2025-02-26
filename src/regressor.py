import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.models.transformer_model import GraphTransformer
from diffusion.distributions import DistributionProperty
from models.LeftNet import LEFTNet
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete, LEFTLossDiscrete
import src.utils as utils


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


class RegressorDiscrete(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, property_norms, property_norms_val, regressor_train=True):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.args = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = 1
        self.Xdim_output = 0
        self.Edim_output = 0
        self.ydim_output = 4
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.atom_type_to_atomic_number = cfg.dataset.atom_type_to_atomic_number

        self.property_norms = property_norms
        self.property_norms_val = property_norms_val

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
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
                                                              timesteps=cfg.model.diffusion_steps)

        cfg.general.regressor_train=regressor_train
        # if cfg.general.regressor_train:
        self.model2 = LEFTNet(cfg=cfg, property_norms=self.property_norms)
        # else:
        #     self.model2 = LEFTNet(cfg=cfg)
        # Marginal transition model
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        y_marginals = torch.tensor([0.4, 0.1, 0.4, 0.1])
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_marginals=y_marginals)

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.train_loss = MeanSquaredError(squared=True)
        self.train_loss2 = MeanSquaredError(squared=True)
        self.val_loss = MeanAbsoluteError()
        self.val_loss2 = MeanAbsoluteError()
        self.test_loss = MeanAbsoluteError()
        self.best_val_mae = 1e8

        self.val_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for i
                              in range(4)]
        self.test_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for
                               i in range(4)]
        self.target_dict = { i : cfg.model.context[i] for i in range(0, len(cfg.model.context) ) }


    def training_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        context = utils.prepare_context(self.args.model.context, data, self.property_norms, target)
        pred, net_out = self.forward(noisy_data, extra_data, node_mask,
                                                      data.pos, data.batch, context)
        # pred = self.forward(noisy_data, extra_data, node_mask)


        mse = self.compute_train_loss(pred, net_out, target, log=i % self.log_every_steps == 0)
        return {'loss': mse}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.train.lr, amsgrad=True, weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                # "monitor": "val_loss",
            },
        }

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.args.general.regressor:
            self.prop_dist = DistributionProperty(self.trainer.datamodule.train_dataloader(),
                                                {k: v for k,v in self.dataset_info.prop2idx.items() if k == self.args.model.context[0]})
            self.prop_dist.set_normalizer(self.property_norms)
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.args)


    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_loss2.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        train_mse = self.train_loss.compute() + self.train_loss2.compute()

        to_log = {"train_epoch/mse": train_mse}
        print(f"Epoch {self.current_epoch}: train_mse: {train_mse :.3f} -- {time.time() - self.start_epoch_time:.1f}s ")

        wandb.log(to_log)
        self.train_loss.reset()
        self.train_loss2.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_loss2.reset()
        reset_metrics(self.val_loss_each)

    def validation_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        # pred = self.forward(noisy_data, extra_data, node_mask)
        context = utils.prepare_context(self.args.model.context, data, self.property_norms, target)
        pred, net_out = self.forward(noisy_data, extra_data, node_mask,
                                                      data.pos, data.batch, context)
        mae = self.compute_val_loss(pred, net_out, target)
        # self.log('val_loss', mae, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': mae}

    def on_validation_epoch_end(self) -> None:
        val_mae = self.val_loss.compute() + self.val_loss2.compute()
        to_log = {"val/epoch_mae": val_mae}
        print(f"Epoch {self.current_epoch}: val_mae: {val_mae :.3f}")
        wandb.log(to_log)
        self.log('val/epoch_mae', val_mae, on_epoch=True, on_step=False)

        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_mae, self.best_val_mae))


        print('Val loss each target:')
        for i in range(4):
            mae_each = self.val_loss_each[i].compute()
            print(f"Target {self.target_dict[i]}: val_mae: {mae_each :.3f}")
            to_log_each = {f"val_epoch/{self.target_dict[i]}_mae": mae_each}
            wandb.log(to_log_each)

        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def on_test_epoch_start(self) -> None:
        self.test_loss.reset()
        reset_metrics(self.test_loss_each)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        time_step = torch.randint(lowest_t, self.T, size=(X.size(0) // 2 + 1,), device=X.device)  # (bs, 1)
        time_step = torch.cat(
            [time_step, self.T - time_step - 1], dim=0)[:X.size(0)]
        t_int = time_step.unsqueeze(1).long()
        # t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim)
        # assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'time_step': time_step, 'X_t': z_t.X, 'E_t': z_t.E,
                      'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, net_out, target):
        """Computes MAE.
           pred: (batch_size, n, total_features)
           target: (batch_size, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """

        for i in range(pred.y.shape[1]):
            mae_each = self.val_loss_each[i](pred.y[:, i], target[:, i])

        mae1 = self.val_loss(pred.y, target)
        mae2 = self.val_loss2(net_out, target[..., :1])
        mae = mae1 + mae2
        return mae

    def forward(self, noisy_data, extra_data, node_mask,
                pos=None, batch=None, context=None):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        output1 = self.model1(X, E, y, node_mask)
        if pos == None:
            output2, atomic_t, pos_t = None, None, None
        else:
            atomic_t, pos_t = self.left_process(pos, batch, X, noisy_data['beta_t'], node_mask)
            output2 = self.model2(z=atomic_t,
                                  node_mask=node_mask,
                                  pos_perturbed=pos_t,
                                  batch=batch,
                                  context=context,
                                  time_step=noisy_data['time_step'])
        return output1, output2

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']

        assert extra_X.shape[-1] == 0, 'The regressor model should not be used with extra features'
        assert extra_E.shape[-1] == 0, 'The regressor model should not be used with extra features'
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)

    def compute_train_loss(self, pred, net_out, target, log: bool):
        """
           pred: (batch_size, n, total_features)
               pred_epsX: bs, n, dx
               pred_epsy: bs, n, n, dy
               pred_eps_z: bs, dz
           data: dict
           noisy_data: dict
           Output: mse (size 1)
       """
        mse1 = self.train_loss(pred.y, target)
        mse2 = self.train_loss2(net_out, target[...,:1])
        mse = mse1 + mse2

        if log:
            wandb.log({"train_loss/batch_mse": mse.item()}, commit=True)
        return mse
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
        pos_perturbed = pos[node_mask] + utils.center_pos(pos_noise[node_mask], batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()  #batch, 3
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