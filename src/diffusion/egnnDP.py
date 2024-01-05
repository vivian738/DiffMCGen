import math
import torch
from src.models.egnn import EGNNSparseNetwork
from src.diffusion.edge import (get_edge_encoder, MultiLayerPerceptron, _extend_to_radius_graph,
                                assemble_atom_pair_feature)
from src.diffusion.diffusion_utils import get_timestep_embedding, get_beta_schedule
from src.diffusion.schnet import SchNetEncoder
from torch_scatter import scatter_add, scatter_mean


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.type_as(x)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GlobalEdgeEGNN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.edge_encoder_global = get_edge_encoder(cfg)
        self.training = cfg.model.train
        self.context = cfg.model.context
        self.atom_type_input_dim = cfg.model.num_atom
        self.num_convs = cfg.model.num_convs
        self.hidden_dim = cfg.model.hidden_dim
        self.atom_out_dim = cfg.model.num_atom  # contains charge or not
        self.soft_edge = cfg.model.soft_edge
        self.norm_coors = cfg.model.norm_coors
        self.time_emb = True
        self.cutoff=cfg.model.cutoff
        self.encoder_global = EGNNSparseNetwork(
            n_layers=self.num_convs,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=self.hidden_dim,
            edge_attr_dim=self.hidden_dim,
            m_dim=self.hidden_dim,
            soft_edge=self.soft_edge,
            norm_coors=self.norm_coors
        )

        self.context_encoder = SchNetEncoder(
            hidden_channels=cfg.model.hidden_dim,
            num_filters=cfg.model.hidden_dim,
            num_interactions=cfg.model.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=10,  # config.cutoff
            smooth=cfg.model.smooth_conv,
            input_dim=self.atom_type_input_dim,
            time_emb=False,
            context=True
        )
        self.atom_type_input_dim = self.atom_type_input_dim * 2
        ctx_nf = len(self.context)
        self.atom_type_input_dim = self.atom_type_input_dim + ctx_nf

        self.temb = torch.nn.Module()
        self.temb.dense = torch.nn.ModuleList([
            torch.nn.Linear(cfg.model.hidden_dim,
                            cfg.model.hidden_dim * 4),
            torch.nn.Linear(cfg.model.hidden_dim * 4,
                            cfg.model.hidden_dim * 4),
        ])
        self.temb_proj = torch.nn.Linear(cfg.model.hidden_dim * 4,
                                         cfg.model.hidden_dim)

        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * cfg.model.hidden_dim,
            [cfg.model.hidden_dim, cfg.model.hidden_dim // 2, 1],
            activation=cfg.model.mlp_act
        )
        self.grad_global_node_mlp = MultiLayerPerceptron(
            1 * cfg.model.hidden_dim,
            [cfg.model.hidden_dim, cfg.model.hidden_dim // 2, self.atom_out_dim],
            activation=cfg.model.mlp_act
        )
        self.model_global = torch.nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])

        betas = get_beta_schedule(
            beta_schedule=cfg.model.beta_schedule,
            beta_start=cfg.model.beta_start,
            beta_end=cfg.model.beta_end,
            num_diffusion_timesteps=cfg.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = torch.nn.Parameter(betas, requires_grad=False)
        # variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = torch.nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def net(self, atom_type, pos, bond_index, bond_type, batch, time_step,
                edge_type=None, edge_index=None, edge_length=None, context=None, vae_noise=None):
        if self.training:
            edge_length = get_distance(pos, bond_index).unsqueeze(-1)
            m, log_var = self.context_encoder(
                z=atom_type,
                edge_index=bond_index,
                edge_length=edge_length,
                edge_attr=None,
                embed_node=False  # default is True
            )
            std = torch.exp(log_var * 0.5)
            z = torch.randn_like(log_var)
            ctx = m + std * z
            atom_type = torch.cat([atom_type, ctx], dim=1)
            kl_loss = 0.5 * torch.sum(torch.exp(log_var) + m ** 2 - 1. - log_var)

        else:
            ctx = vae_noise
            atom_type = torch.cat([atom_type, ctx], dim=1)
            kl_loss = 0

        if len(self.context) > 0 and self.context is not None:
            atom_type = torch.cat([atom_type, context], dim=1)

        if self.time_emb:
            nonlinearity = torch.nn.ReLU()
            temb = get_timestep_embedding(time_step, self.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            atom_type = torch.cat([atom_type, temb.index_select(0, batch)], dim=1)

        if edge_index is None or edge_type is None or edge_length is None:
            bond_type = torch.ones(bond_index.size(1), dtype=torch.long).to(bond_index.device)
            edge_index, edge_type = _extend_to_radius_graph(
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                cutoff=self.cutoff,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
        local_edge_mask = is_radius_edge(edge_type)
        # Emb time_step for edge
        if self.time_emb:
            node2graph = batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            temb_edge = temb.index_select(0, edge2graph)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )  # Embed edges
        if self.time_emb:
            edge_attr_global += temb_edge
        # EGNN
        node_attr_global, _ = self.encoder_global(
            z=atom_type,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
            batch=batch
        )
        """
        Assemble pairwise features
        """
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )  # (E_global, 2H)
        """
        Invariant features of edges (radius graph, global)
        """
        dist_score_global = self.grad_global_dist_mlp(h_pair_global)  # (E_global, 1)
        node_score_global = self.grad_global_node_mlp(node_attr_global)

        return dist_score_global, node_score_global, edge_index, edge_type, edge_length, local_edge_mask, kl_loss

    def forward(self, batch, context=None, is_sidechain=None):
        atom_type = batch.atom_feat_full.float()
        pos = batch.pos
        bond_index = batch.edge_index
        bond_type = batch.edge_type
        batch = batch.batch

        # N = atom_type.size(0)
        node2graph = batch
        num_graphs = node2graph[-1] + 1

        # Sample time step
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device)

        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]

        a = self.alphas.index_select(0, time_step)  # (G, )

        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

        """
        Independently
        - Perturb pos
        """
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + center_pos(pos_noise, batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # pos_perturbed = a_pos.sqrt()*pos+(1.0 - a_pos).sqrt()*center_pos(pos_noise,batch)
        """
        Perturb atom
        """
        atom_noise = torch.zeros(size=atom_type.size(), device=atom_type.device)
        atom_noise.normal_()
        atom_type = torch.cat([atom_type[:, :-1] / 4, atom_type[:, -1:] / 10], dim=1)
        atom_perturbed = a_pos.sqrt() * atom_type + (1.0 - a_pos).sqrt() * atom_noise

        vae_noise = torch.randn_like(atom_type)  # N(0,1)
        # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3) # clip N(0,1)
        # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device) # N(0,3)
        # vae_noise = torch.zeros_like(atom_type).uniform_(-1,+1) # U(-1,1)
        net_out = self.net(
            atom_type=atom_perturbed,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            time_step=time_step,
            context=context,
            vae_noise=vae_noise
        )  # (E_global, 1), (E_local, 1)
        dist_score_global, node_score_global, edge_index, _, edge_length, local_edge_mask = net_out[:-1]
        edge2graph = node2graph.index_select(0, edge_index[0])

        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))

        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction

        global_mask = torch.logical_and(
            torch.logical_or(torch.logical_and(d_perturbed > self.cutoff, d_perturbed <= 10),
                             local_edge_mask.unsqueeze(-1)),
            ~local_edge_mask.unsqueeze(-1)
        )

        edge_inv_global = torch.where(global_mask, dist_score_global, torch.zeros_like(dist_score_global))
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)

        # global pos
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 1 * torch.sum(loss_global, dim=-1, keepdim=True)

        loss_node_global = (node_score_global - atom_noise) ** 2
        loss_node_global = 1 * torch.sum(loss_node_global, dim=-1, keepdim=True)

        # loss for atomic eps regression

        vae_kl_loss = net_out[-1]
        loss = loss_global + loss_node_global + vae_kl_loss

        return loss

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
    #     + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
    return score_pos


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def is_radius_edge(edge_type):
    return edge_type == 0