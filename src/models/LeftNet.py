import math
from math import pi
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import Embedding

from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_mean
from models.layers import PositionsMLP
from torch_geometric.utils import to_dense_batch
from diffusion.diffusion_utils import get_timestep_embedding

def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))

def swish(x):
    return x * torch.sigmoid(x)

class MappingBlock(nn.Module):
    def __init__(self, atom_type_to_atomic_number):
        super().__init__()
        self.atom_type_to_atomic_number = atom_type_to_atomic_number

        mapping_tensor = torch.full(
            (len(atom_type_to_atomic_number),),
            -1.0,
            dtype=torch.float32
        )
        for key, value in atom_type_to_atomic_number.items():
            mapping_tensor[key] = float(value)

        self.mapping_tensor = nn.Parameter(mapping_tensor)

    def forward(self, node_gt):
        atomic_gt = torch.gather(
            self.mapping_tensor.unsqueeze(0),
            1,
            node_gt.long().unsqueeze(0)
        ).squeeze(0)
        return atomic_gt

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
            nn.LayerNorm(hid_dim, elementwise_affine=False),
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
        self.T = cfg.model.diffusion_steps
        self.hidden_channels = cfg.model.hidden_channels
        self.cutoff = cfg.model.cutoff
        self.pos_require_grad = cfg.model.pos_require_grad
        self.num_radial = cfg.model.num_radial
        self.atomic_input_dim = cfg.model.num_atomic + 1
        self.atom_type_to_atomic_number = cfg.dataset.atom_type_to_atomic_number
        # self.mapping_block = MappingBlock(self.atom_type_to_atomic_number)
        
        self.z_emb = Embedding(self.atomic_input_dim, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_channels, len(self.atom_type_to_atomic_number))
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
        # self.mlp_in_pos = PositionsMLP(cfg.model.hidden_mlp_dims['pos'])
        # self.mlp_out_pos = PositionsMLP(cfg.model.hidden_mlp_dims['pos'])
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hidden_channels,
                            self.hidden_channels * 4),
            torch.nn.Linear(self.hidden_channels * 4,
                            self.hidden_channels * 4),
        ])
        self.temb_proj = torch.nn.Linear(self.hidden_channels * 4, self.hidden_channels)
        self.combined_proj = nn.Linear(3, 1)
        if self.pos_require_grad:
            self.out_forces = EquiOutput(self.hidden_channels)
        
        # for node-wise frame
        self.mean_neighbor_pos = aggregate_pos(aggr='mean')

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        # self.y_mean = property_norms[cfg.model.context[0]]['mean']
        # self.y_std = property_norms[cfg.model.context[0]]['mad']

        self.reset_parameters()

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
    

    def forward(self, z, node_mask, pos_perturbed, batch, context, time_step):
        # pos_perturbed = self.mlp_in_pos(noise_pos, node_mask)[node_mask]

        # z = z[node_mask].long()
        # time_step = time_step / self.T
        time_emb = time_step.index_select(0, batch)
        temb = get_timestep_embedding(time_emb, self.hidden_channels)  #batch, hidden_channels
        temb = self.temb.dense[0](temb)
        temb = nn.ReLU()(temb)
        temb = self.temb.dense[1](temb)
        temb = self.temb_proj(nn.ReLU()(temb))  # (G, dim)
        # embed z
        z_ = self.z_emb(z) # (N, bs)
        if context is None:
            ctx = torch.zeros_like(z.float()).uniform_(-1, +1).unsqueeze(1).unsqueeze(-1)  # G,
            z_tc = torch.cat([temb.index_select(0, batch).unsqueeze(-1), ctx.expand(-1, 64, -1)], dim=-1)
        else:
            ctx = ((context - context.min()) / (context.max() - context.min())).unsqueeze(-1)
            z_tc = torch.cat([temb.index_select(0, batch).unsqueeze(-1), ctx.expand(-1, 64, -1)], dim=-1)
        
        z_emb = self.combined_proj(torch.cat([z_.unsqueeze(-1), z_tc], dim=-1)).squeeze(-1) # (N, bs)

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
        # edge_diff += temb
        edge_cross = torch.cross(pos_perturbed[i], pos_perturbed[j])
        edge_cross = _normalize(edge_cross)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # edge_frame shape: (num_edges, 3, 3)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        
        del edge_cross, edge_vertical
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        perm1 = torch.permute(scalrization1, (0, 2, 1))
        scalar3 = self.lin(perm1).add_(perm1[:, :, 0].unsqueeze(2)).squeeze(-1)

        perm2 = torch.permute(scalrization2, (0, 2, 1))
        scalar4 = self.lin(perm2).add_(perm2[:, :, 0].unsqueeze(2)).squeeze(-1)


        # scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + torch.permute(scalrization1, (0, 2, 1))[:, :,
        #                                                                 0].unsqueeze(2)).squeeze(-1)
        # scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + torch.permute(scalrization2, (0, 2, 1))[:, :,
        #                                                                 0].unsqueeze(2)).squeeze(-1)
        
        A_i_j = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)
        A_i_j = torch.cat((A_i_j, radial_hidden, radial_emb), dim=-1)
        
        del scalar3, scalar4, scalrization1, scalrization2, S_i_j, perm1, perm2
        torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()

        if self.pos_require_grad:
            _, pos_gt = self.out_forces(s, vec)

        h = self.embedding_out(s)
        node_gt = torch.nn.functional.gumbel_softmax(h, tau=1, hard=True, dim=-1)
        # node_gt = torch.sum(node_gt * torch.arange(node_gt.size(-1), device=h.device), dim=-1)
        # atomic_gt = self.mapping_block(node_gt)
        atomic_gt, _ = to_dense_batch(x=node_gt, batch=batch)
        h = atomic_gt * node_mask.unsqueeze(-1)  #(bs, n, dx)
        # h = h * self.y_std + self.y_mean
        # dpos_gt= pos_perturbed + pos_gt
        pos_gt, _ = to_dense_batch(x=pos_gt, batch=batch)
        # pos_gt = self.mlp_out_pos(pos_gt, node_mask)
        
        pos = pos_gt * node_mask.unsqueeze(-1)  #(bs, n, 3)
    
        return h, pos

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center