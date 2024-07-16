import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter_add, scatter_mean
import torch
import omegaconf
import wandb
from torch_geometric.utils import dense_to_sparse
import numpy as np
import torch.nn.functional as F


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch, y):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    # y = torch.stack(y, dim=0)

    return PlaceHolder(X=X, E=E, y=y), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.to(x.device)
        return self

    def mask(self, node_mask, collapse=False):
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        # diag_mask = ~torch.eye(n, dtype=torch.bool,
        #                        device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2 # * diag_mask
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'entity': 'vicky863',
              'config': config_dict, 'settings': wandb.Settings(_disable_stats=True), 'reinit': True,
              'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

def prepare_context(conditioning, minibatch, property_norms):
    # batch_size = minibatch['batch'][-1] + 1
    context_node_nf = 0
    context_list = []
    # for i, key in enumerate(conditioning):
    key=conditioning[0]
    properties = minibatch.y[..., 0]
    properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
    if len(properties.size()) == 1:
        # Global feature.
        # assert properties.size() == (batch_size,)
        properties = properties.index_select(0, minibatch['batch'])
        context_list.append(properties.unsqueeze(1))
        context_node_nf += 1
    elif len(properties.size()) == 2 or len(properties.size()) == 3:
        # Node feature.
        # assert properties.size(0) == batch_size

        context_key = properties

        # Inflate if necessary.
        if len(properties.size()) == 2:
            context_key = context_key.unsqueeze(2)

        context_list.append(context_key)
        context_node_nf += context_key.size(2)
    else:
        raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=1)
    # Mask disabled nodes!
    assert context.size(1) == context_node_nf
    return context


def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def normalize_prop(y):
    first_col = y[:, 0]
    y[:, 0] = (first_col - first_col.min()) / (first_col.max() - first_col.min())

    second_col = y[:, 1]
    y[:, 1] = (second_col - second_col.min()) / (second_col.max() - second_col.min())
    normalized_y = y
    return normalized_y


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def padding(data_list, max_nodes):
    padding_list = []
    for data in data_list:
        if data.size(0)<=max_nodes:
            padding_data = torch.cat([data,torch.zeros(max_nodes-data.size(0), data.size(1)).to(data.device)],dim=0)
            padding_list.append(padding_data)
    return torch.stack(padding_list)


def ex_batch(X, E, node_mask):
    atom_exist = torch.argmax(X, dim=-1)
    # weighted_tensor = node_type[atom_exist]
    atom_type = torch.cat([X[node_mask], atom_exist[node_mask].unsqueeze(1)], dim=1)
    edge_exist = torch.argmax(E, dim=-1) #[B. N. N]
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    adj = edge_exist * edge_mask
    edge_index, edge_attr = dense_to_sparse(adj, node_mask)
    # edge_attr = edge_type[edge_attr]
    return atom_type, edge_index, edge_attr