import math
from math import pi
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import Embedding

from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_mean
from diffusion.diffusion_utils import get_timestep_embedding, get_beta_schedule
from diffusion.layers import get_distance

def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))

def swish(x):
    return x * torch.sigmoid(x)

## radial basis function to embed distances
class rbf_emb(nn.Module):
    def __init__(self, num_rbf, soft_cutoff_upper, rbf_trainable=False):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist=dist.unsqueeze(-1)
        soft_cutoff = 0.5 * \
                  (torch.cos(dist * pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff*torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    def __init__(self, atomic_input_dim, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(atomic_input_dim, hid_dim)
        self.hid_dim = hid_dim

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.embedding(z)
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j

class S_vector(MessagePassing):
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU())

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_radial,
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.inv_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3), nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3), )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector):
        xh = self.x_proj(x)

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.inv_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.equi_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.equi_proj.weight)
        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, node_frame):

        vec = self.equi_proj(vec)
        vec1,vec2 = torch.split(
                 vec, self.hidden_channels, dim=-1
             )

        scalrization = torch.sum(vec1.unsqueeze(2) * node_frame.unsqueeze(-1), dim=1)
        scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = torch.norm(vec1, dim=-2) # torch.sqrt(torch.sum(vec1 ** 2, dim=-2))
        
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xequi_proj(
            torch.cat(
                [x, scalar], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class aggregate_pos(MessagePassing):

    def __init__(self, aggr='mean'):  
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class EquiOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x.squeeze(), vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


class LEFTNet(torch.nn.Module):
    r"""
        LEFTNet

        Args:
            pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output with respect to the atomic positions. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`32`)
            y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
            y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)

    """

    def __init__(self, cfg):
        super(LEFTNet, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.hidden_channels = cfg.model.hidden_channels
        self.cutoff = cfg.model.cutoff
        self.pos_require_grad = cfg.model.pos_require_grad
        self.num_radial = cfg.model.num_radial
        self.atomic_input_dim = cfg.model.num_atomic * 2 + 1
        self.atom_type_to_atomic_number = cfg.dataset.atom_type_to_atomic_number

        # self.temb = torch.nn.Module()
        # self.temb.dense = torch.nn.ModuleList([
        #     torch.nn.Linear(self.hidden_channels,
        #                     self.hidden_channels * 4),
        #     torch.nn.Linear(self.hidden_channels * 4,
        #                     self.hidden_channels * 4),
        # ])
        # self.temb_proj = torch.nn.Linear(self.hidden_channels * 4,
        #                                     self.hidden_channels)
        
        self.z_emb = Embedding(self.atomic_input_dim, self.hidden_channels)
        self.radial_emb = rbf_emb(self.num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(self.num_radial, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        
        self.neighbor_emb = NeighborEmb(self.atomic_input_dim, self.hidden_channels)

        self.S_vector = S_vector(self.hidden_channels)

        self.lin = nn.Sequential(
            nn.Linear(3, self.hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels // 4, 1))

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.message_layers.append(
                EquiMessagePassing(self.hidden_channels, self.num_radial).jittable()
            )
            self.FTEs.append(FTE(self.hidden_channels))

        # self.last_layer = nn.Linear(self.hidden_channels, 1)
        if self.pos_require_grad:
            self.out_forces = EquiOutput(self.hidden_channels)
        
        # for node-wise frame
        self.mean_neighbor_pos = aggregate_pos(aggr='mean')

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        # self.fc_gamma = nn.Linear(1, self.hidden_channels)
        # self.fc_beta = nn.Linear(1, self.hidden_channels)

        self.reset_parameters()

        betas = get_beta_schedule(
            beta_schedule=cfg.model.beta_schedule,
            beta_start=cfg.model.beta_start,
            beta_end=cfg.model.beta_end,
            num_diffusion_timesteps=cfg.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = torch.nn.Parameter(betas, requires_grad=False)
        # # variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = torch.nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        # self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def egn_process(self, pos, batch):

        # N = atom_type.size(0)
        node2graph = batch
        num_graphs = node2graph[-1] + 1

        # # Sample time step
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device)
        
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step.long())  # (G, )

        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

        """
        Independently
        - Perturb pos
        """
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + center_pos(pos_noise, batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # pos_perturbed = torch.nan_to_num(pos_perturbed, nan=0, posinf=1e9, neginf=-1e9)
        # pos_perturbed = a_pos.sqrt()*pos+(1.0 - a_pos).sqrt()*center_pos(pos_noise,batch)

        return pos_perturbed

    def forward(self, z, pos_perturbed, batch, context):
        # if self.pos_require_grad:
        #     pos_perturbed.requires_grad_()
        # embed time_step for node
        # fc_gamma = self.fc_gamma(context)
        # fc_beta = self.fc_beta(context)

        # atomic_nc = fc_gamma * z + fc_beta
        context_normal = (context - context.min()) / (context.max() - context.min())
        atomic_nc = z * (1 + context_normal.squeeze())
        # nonlinearity = torch.nn.ReLU()
        # temb = get_timestep_embedding(time_step, self.hidden_channels)
        # temb = self.temb.dense[0](temb)
        # temb = nonlinearity(temb)
        # temb = self.temb.dense[1](temb)
        # temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
        # embed z
        # z_, temb = atomic_nc.long(), temb.index_select(0, batch)
        z_emb = self.z_emb(atomic_nc.long())   # (N, bs)
        
        # construct edges based on the cutoff value
        edge_index = radius_graph(pos_perturbed, r=self.cutoff, batch=batch)
        i, j = edge_index
        
        # embed pair-wise distance
        dist = torch.norm(pos_perturbed[i]-pos_perturbed[j], dim=-1)
        # radial_emb shape: (num_edges, num_radial), radial_hidden shape: (num_edges, hidden_channels)
        radial_emb = self.radial_emb(dist)	
        radial_hidden = self.radial_lin(radial_emb)	
        soft_cutoff = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = soft_cutoff.unsqueeze(-1) * radial_hidden

        # init invariant node features
        # shape: (num_nodes, hidden_channels)
        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)

        # init equivariant node features
        # shape: (num_nodes, 3, hidden_channels)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        # bulid edge-wise frame
        edge_diff = pos_perturbed[i] - pos_perturbed[j]
        edge_diff = _normalize(edge_diff)
        edge_cross = torch.cross(pos_perturbed[i], pos_perturbed[j])
        edge_cross = _normalize(edge_cross)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # edge_frame shape: (num_edges, 3, 3)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        
        del edge_cross, edge_vertical
        # build node-wise frame
        mean_neighbor_pos = self.mean_neighbor_pos(pos_perturbed, edge_index)
        node_diff = pos_perturbed - mean_neighbor_pos
        node_diff = _normalize(node_diff)
        node_cross = torch.cross(pos_perturbed, mean_neighbor_pos)
        node_cross = _normalize(node_cross)
        node_vertical = torch.cross(node_diff, node_cross)
        # node_frame shape: (num_nodes, 3, 3)
        node_frame = torch.cat((node_diff.unsqueeze(-1), node_cross.unsqueeze(-1), node_vertical.unsqueeze(-1)), dim=-1)

        del node_cross, node_diff, node_vertical
        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + torch.permute(scalrization1, (0, 2, 1))[:, :,
                                                                        0].unsqueeze(2)).squeeze(-1)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + torch.permute(scalrization2, (0, 2, 1))[:, :,
                                                                        0].unsqueeze(2)).squeeze(-1)
        
        A_i_j = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)
        A_i_j = torch.cat((A_i_j, radial_hidden, radial_emb), dim=-1)
        
        del scalar3, scalar4, scalrization1, scalrization2, S_i_j
        for i in range(self.num_layers):
            # equivariant message passing
            ds, dvec = self.message_layers[i](
                s, vec, edge_index, radial_emb, A_i_j, edge_diff
            )

            s = s + ds
            vec = vec + dvec

            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec, node_frame)
            s = s + ds
            vec = vec + dvec

        if self.pos_require_grad:
            node_gt, pos_gt = self.out_forces(s, vec)

        # s = self.last_layer(s).squeeze(1)
        # node_ft = scatter(s, batch, dim=0)
    
        if self.pos_require_grad:
            return z + node_gt, pos_perturbed + pos_gt
        return s, vec
    
    def sampled_dynamics_pos(self, net_out, pos, batch, i, j, t):
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        node_gt, pos_gt = net_out
        noise = center_pos(0.5 * torch.randn_like(pos) + 
                           0.5 * torch.distributions.Laplace(0, 1).sample(pos.shape).to(pos.device), batch)
        b = self.betas
        t = t[0]
        next_t = (torch.ones(1) * j).to(pos.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        beta_t = 1 - at / at_next
        e = -pos_gt

        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
        mean = (
            (at_next.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * pos
                ) / (1.0 - at)
        mask = 1 - (t == 0).float()
        logvar = beta_t.log()
        pos_next = mean + mask * torch.exp(
            0.5 * logvar) * noise   
        
        return pos_next

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center