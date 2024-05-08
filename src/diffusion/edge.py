import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from rdkit.Chem.rdchem import BondType as BT

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}

class GaussianSmearingEdgeEncoder(Module):

    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        #self.NUM_BOND_TYPES = 22
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians)    # Larger `stop` to encode more cases
        self.bond_emb = Embedding(100, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = torch.cat([self.rbf(edge_length), self.bond_emb(edge_type)], dim=1)
        return edge_attr


class MLPEdgeEncoder(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_type = torch.clamp(edge_type, 0, self.bond_emb.num_embeddings - 1)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)


def get_edge_encoder(cfg):
    if cfg.model.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.model.hidden_dim, cfg.model.mlp_act)
    elif cfg.model.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(cfg.model.hidden_dim // 2, cutoff=cfg.model.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % cfg.model.edge_encoder)
        

class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    if edge_attr is not None:
        h_pair = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (E, 2H)
    else:
        h_pair = torch.cat([h_row, h_col], dim=-1)  # (E, 2H)
    return h_pair


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index, edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)  # (G, )
    node_offset = num_cum_nodes - num_nodes_per_graph  # (G, )
    edge_offset = node_offset[edge2graph]  # (E, )

    num_nodes_square = num_nodes_per_graph ** 2  # (G, )
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (G, )
    edge_start = num_nodes_square_cumsum - num_nodes_square  # (G, )
    edge_start = edge_start[edge2graph]

    all_len = num_nodes_square_cumsum[-1]

    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = node_index.max(dim=-1)[0]
    node_small = node_index.min(dim=-1)[0]
    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

    symm_noise = torch.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (E, 1)
    return d_noise


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    # _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data

    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added.
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type


def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=-1):
    if edge_type.dim() == 1:
        pass
    else:
        edge_type = torch.argmax(edge_type, dim=1)
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(
        edge_index,
        edge_type,
        torch.Size([N, N])
    )

    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    return new_edge_index, new_edge_type



def coarse_grain(pos, node_attr, subgraph_index, batch):
    cluster_pos = scatter_mean(pos, index=subgraph_index, dim=0)  # (num_clusters, 3)
    cluster_attr = scatter_add(node_attr, index=subgraph_index, dim=0)  # (num_clusters, H)
    cluster_batch, _ = scatter_max(batch, index=subgraph_index, dim=0)  # (num_clusters, )

    return cluster_pos, cluster_attr, cluster_batch


def batch_to_natoms(batch):
    return scatter_add(torch.ones_like(batch), index=batch, dim=0)


def get_complete_graph(natoms):
    """
    Args:
        natoms: Number of nodes per graph, (B, 1).
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        num_edges:  (B, ), number of edges per graph.
    """
    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms  # Number of edges per graph

    return edge_index, num_edges