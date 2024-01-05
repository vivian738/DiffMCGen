# from torch_geometric.transforms import Compose, ToDevice
from .qm9_dataset import QM9Dataset
# from .geom_dataset import GeomDrugDataset
# from .zinc_dataset import ZincDataset
from .moses_dataset import MOSESDataset
# from .datasets_config import get_dataset_info
# from torch_geometric.data import Data
# from torch.utils.data import DataLoader
# import torch
import pathlib
from rdkit.Chem.rdchem import BondType as BT
import os
# import pandas as pd
import numpy as np
# import pickle
# from scipy.spatial import distance_matrix
# import multiprocessing
# from itertools import repeat
# import networkx as nx
import torch
from torch.utils.data import DataLoader
from rdkit.Chem import ChemicalFeatures
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Data
import warnings
from rdkit import RDConfig

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')


BOND_TYPES = {
    BT.UNSPECIFIED: 0,
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
ATOM_SYMBOLS= {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'Br':8, 'I':9}
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
AROMATIC = {'True':0, 'False':1}

def atom_features(mol):
    # pharmacophore information
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    Num = mol.GetNumAtoms()
    phar_mat = np.zeros([Num, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(mol):
        phar_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    results = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        hyb = HYBRIDIZATION_TYPE_ID[str(atom.GetHybridization())]
        sym = ATOM_SYMBOLS[atom.GetSymbol()]
        deg = atom.GetDegree()
        vale = atom.GetImplicitValence()
        aro = AROMATIC[str(atom.GetIsAromatic())]
        results.append((idx, [sym, hyb, deg, vale, aro]))
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    results = sorted(results)
    results = [v[1] for v in results]
    atom_feats = np.array(results).astype(np.float32)

    ptable = Chem.GetPeriodicTable()
    pos = np.array(mol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(Num):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int)

    return phar_mat, atom_feats, pos, center_of_mass, element

        # graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def edge_feature(mol):
    row, col, edge_type = [], [], []
    Num = mol.GetNumAtoms()

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

        # graph.add_edge(start, end, feats=torch.from_numpy(bond_feats))
    perm = (edge_index[0] * Num + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    return edge_index, edge_type


def ligand2dict(path):
    # graph = nx.Graph()  #无向图
    # Remove Hydrogens
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError

    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    phar_mat, atom_feats, pos, center_of_mass, element = atom_features(rdmol)
    edge_index, edge_type = edge_feature(rdmol)
    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': atom_feats,
        'phar_match': phar_mat
    }
    # graph = graph.to_directed()
    # x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    # edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return data

########################QM9常用方法######################
def get_dataset(data_name, cfg, transform=True):
    """Create dataset for training and evaluation."""

    # Obtain dataset info
    # dataset_info = get_dataset_info(config.data.info_name)

    # get transform

    if transform:
        prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11, 'pharmo_score':16, 'SA':17, 'QED':18}
        atom_encoder= {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'P': 6, 'Cl':7, 'Br':8, 'I':9}
        name_transform = 'EdgeComCondMulti'
        if name_transform == 'EdgeComCond':

            transform = EdgeComCondTransform(atom_encoder.values(), include_aromatic=True,
                                             property_idx=prop2idx['gap'])
        elif name_transform == 'EdgeComCondMulti':

            transform = EdgeComCondMultiTransform(atom_encoder.values(), include_aromatic=True,
                                                  property_idx1=prop2idx['gap'], property_idx2=prop2idx['pharmo_score'], property_idx3=prop2idx['SA'], property_idx4=prop2idx['QED'])
        else:
            raise ValueError('Invalid data transform name')
    else:
        transform = None
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    root_path = os.path.join(base_path, cfg.dataset.datadir)

    # Build up dataset
    if data_name == 'QM9':
        dataset = QM9Dataset(root_path, transform=transform)
    # elif config.data.name == 'GeomDrug':
    #     dataset = GeomDrugDataset(config.data.root, config.data.processed_file, transform=transform)
    # elif config.data.name == 'Zinc250k':
    #     dataset = ZincDataset(config.data.root, transform=transform)
    elif data_name == 'MOSES':
        dataset = MOSESDataset(root_path, transform=transform)
    else:
        raise ValueError('Undefined dataset name.')

    # Split dataset
    split_idx = dataset.get_cond_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    # second_train_dataset = dataset.index_select(split_idx['second_train'])
    val_dataset = dataset.index_select(split_idx['valid'])
    test_dataset = dataset.index_select(split_idx['test'])
    return train_dataset, val_dataset, test_dataset, prop2idx




def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# setup dataloader
def get_dataloader(train_ds, val_ds, test_ds):
    # choose collate_fn
    collate_fn = eval('collate_cond') # collate_property_classifier, collate_edge_2D, collate_edge, collate_node

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                              num_workers=16, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                            num_workers=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=16, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# transform data

class EdgeComCondTransform(object):
    """
    Transform data with node features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx = property_idx

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

            data.property = properties[0, self.property_idx:self.property_idx+1] - \
                            torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            property = properties[0, self.property_idx:self.property_idx+1]
            data.property = property

        return data


class EdgeComCondMultiTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx1, property_idx2, property_idx3, property_idx4):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx1 = property_idx1
        self.property_idx2 = property_idx2
        self.property_idx3 = property_idx3
        self.property_idx4 = property_idx4

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        prop_list = [self.property_idx1, self.property_idx2, self.property_idx3, self.property_idx4]
        property_data = []
        for prop_idx in prop_list:
            if prop_idx == 11:
                Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
                atom_types = data.atom_type
                atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

                property_data.append(properties[0, prop_idx:prop_idx+1] - \
                                torch.sum((atom_counts * torch.tensor(Cv_atomref))))
            else:
                property = properties[0, prop_idx:prop_idx+1]
                property_data.append(property)
        data.property = torch.cat(property_data)

        return data


class PropClassifierTransform(object):
    """
        Transform data with node and edge features.
        Conditional property.

    """
    def __init__(self, atom_type_list, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.property_idx = property_idx

    def __call__(self, data: Data):
        data.charge = None
        atom_type = data.atom_type
        one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.one_hot = one_hot.float()
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
            data.property = data.y[0, 11] - torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            data.property = data.y[0, self.property_idx]

        return data


def pad_node_feature(x, pad_len):
    x_len, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)
        new_x[:x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_feature(x, pad_len):
    # x: [N_node, N_node, ch]
    x_len, _, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, pad_len, x_dim])
        new_x[:x_len, :x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_node_mask(node_num, pad_len, dtype):
    node_mask = torch.zeros(pad_len, dtype=dtype)
    node_mask[:node_num] = 1.
    return node_mask.unsqueeze(0)


# collate function: padding with the max node

def collate_node(items):
    items = [(item.one_hot, item.pos, item.fc, item.num_atom) for item in items]
    one_hot, positions, formal_charges, num_atoms = zip(*items)
    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        formal_charges=formal_charges
    )


def collate_edge(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_edge_2D(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_cond(items):
    # collate_fn for the condition generation
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom, item.property) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, property = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # context property
    property = torch.stack(property, dim=0)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        context=property
    )


def collate_property_classifier(items):
    # add conds for the property iterations
    items = [(item.one_hot, item.pos, item.num_atom, item.property) for item in items]
    one_hot, positions, num_atoms, graph_properties = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    property = torch.stack(graph_properties, dim=0)

    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        property=property
    )
