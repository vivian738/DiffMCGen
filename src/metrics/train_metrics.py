import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection, MeanAbsoluteError
import time
import wandb
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    HuberLossMetric, NLL
from torch_scatter import scatter_add
from diffusion.layers import eq_transform, get_distance, is_train_edge


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        # self.cross_active_edge = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, node_mask, pred_y, true_X, true_E, true_y, log: bool): #active_edge_t0, 
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        bs, n = node_mask.shape
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/X_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



class DualLossDiscrete(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.pos_global_loss = 2 * MeanSquaredError()
        # self.pos_local_loss = 5 * MeanSquaredError()
        self.cutoff = cutoff

    def forward(self, net_out, a, pos, pos_perturbed, node2graph, is_sidechain, log: bool):
        edge_inv_global, edge_index, _, edge_length = net_out #edge_inv_local, , local_edge_mask
        edge2graph = node2graph.index_select(0, edge_index[0])

        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction

        # global_mask = torch.logical_and(
        #     torch.logical_or(d_perturbed <= self.cutoff, local_edge_mask.unsqueeze(-1)),
        #     ~local_edge_mask.unsqueeze(-1)
        # )
        # target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        # edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(d_target, pos_perturbed, edge_index, edge_length) #target_d_global
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss = self.pos_global_loss(node_eq_global, target_pos_global) if target_pos_global.numel() > 0 else 0.0

        # target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        # node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        # loss_local = self.pos_local_loss(node_eq_local, target_pos_local) if target_pos_local.numel() >0 else 0.0
        # loss = loss_global + loss_local
        if log:
            to_log = {
                'train_loss/pos_loss': (5 *loss).detach(),
                # 'train_loss/global_pos_MSE': self.pos_global_loss.compute(),
                # 'train_loss/local_pos_MSE': self.pos_local_loss.compute()
                }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return 5 * loss

    def reset(self):
        self.pos_global_loss.reset()
        # self.pos_local_loss.reset()

    def log_epoch_metrics(self):
        global_pos_loss = self.pos_global_loss.compute() if self.pos_global_loss.metric_b.total > 0 else -1
        # local_pos_loss = self.pos_local_loss.compute() if self.pos_local_loss.metric_b.total > 0 else -1
        to_log = {
            "train_epoch/global_pos_loss": global_pos_loss}
            # "train_epoch/local_pos_loss": local_pos_loss}
        if wandb.run:
            wandb.log(to_log)
        return to_log

class LEFTLossDiscrete(nn.Module):
    def __init__(self, atom_type_to_atomic_number):
        super().__init__()
        self.loss_pos = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)
        # self.loss_cond = MeanAbsoluteError()
        self.atom_type_to_atomic_number = atom_type_to_atomic_number
    
    def forward(self, net_out, pos_t, pos, node_mask, log:bool):
        node_gt, pos_gt = net_out
        pos_noise = pos_t - pos
        loss_pos = self.loss_pos(pos_gt[node_mask], pos_noise[node_mask])
        
        # loss_cond = self.loss_cond(node_gt[node_mask], atom_noise[node_mask])
        loss = 5 * loss_pos
        if log:
            to_log = {
                    'train_loss/model2_loss': loss.detach()}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss
    
    def reset(self):
        self.loss_pos.reset()
        # self.loss_cond.reset()

    def log_epoch_metrics(self):
        pos_loss = self.loss_pos.compute() if self.loss_pos.total > 0 else -1
        # cond_loss = self.loss_cond.compute() if self.loss_cond.total > 0 else -1
        
        to_log = {
            "train_epoch/pos_loss": pos_loss}
        if wandb.run:
            wandb.log(to_log)
        return to_log