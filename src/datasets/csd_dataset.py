import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
from torch_geometric.data import Data, InMemoryDataset
from rdkit.Chem import AllChem
from torch_geometric.utils import subgraph

import utils as utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from analysis.rdkit_functions import compute_molecular_metrics

from eval import sascorer
from rdkit.Chem import QED
from numpy.random import RandomState
from datasets.pharmacophore_eval import mol2ppgraph, match_score, get_best_match_phar_models
from torch_geometric.utils import to_dense_batch
import joblib
import multiprocessing
from dgl.data.utils import load_graphs

class EdgeComCondTransform(object):
    """
    Transform data with node features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, atom_index, include_aromatic, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx = property_idx
        self.atom_index = {v : k for k, v in atom_index.items()}

    def __call__(self, data: Data):
        '''
                one-hot feature
        '''
        atom_type = data.atom_type.tolist()
        # print(data.atom_type)
        atom_type = torch.tensor([self.atom_index[i] for i in atom_type])
        # print(atom_type)
        data.atom_feat = F.one_hot(atom_type, num_classes=len(self.atom_index))
        data.atom_feat_full = torch.cat([data.atom_feat, data.atom_type.unsqueeze(1)], dim=1)
        # data.atom_feat_full = torch.cat([data.atom_feat_full, data.charge.unsqueeze(1)], dim=1)
        properties = data.y
        data.y = properties[0, self.property_idx:self.property_idx+1]

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

    def __init__(self, atom_type_list, atom_index, include_aromatic, property_idx1, property_idx2, property_idx3, property_idx4):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.atom_index = {v : k for k, v in atom_index.items()}
        self.property_idx1 = property_idx1
        self.property_idx2 = property_idx2
        self.property_idx3 = property_idx3
        self.property_idx4 = property_idx4

    def __call__(self, data: Data):
        '''
            one-hot feature
        '''
        atom_type = data.atom_type.tolist()
        # print(data.atom_type)
        # atom_type = torch.tensor([self.atom_index[i] for i in atom_type])
        atom_type = torch.tensor(atom_type)
        # print(atom_type)
        data.atom_feat = F.one_hot(atom_type, num_classes=len(self.atom_index))
        data.atom_feat_full = torch.cat([data.atom_feat, data.atom_type.unsqueeze(1)], dim=1)
        # data.atom_feat_full = torch.cat([data.atom_feat_full, data.charge.unsqueeze(1)], dim=1)
        
        properties = data.y
        prop_list = [self.property_idx1, self.property_idx2, self.property_idx3, self.property_idx4]
        property_data = []
        for prop_idx in prop_list:
            property = properties[0, prop_idx]
            property_data.append(property)
        property_ten = torch.tensor(property_data)
        data.y = property_ten.unsqueeze(0)

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
        data.charge = data.charge
        atom_type = data.atom_type
        one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.one_hot = one_hot.float()
        data.y = data.y[0, self.property_idx]

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

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def process_mol(args):
    mol, pp_graph_list, loaded_reg = args
    if mol is None or '.' in Chem.MolToSmiles(mol):
        return (0, 0, 0, 0)  # 如果分子无效，返回默认值
    else:
        pharma_match_score_list = [match_score(mol, pp_graph) for pp_graph in pp_graph_list]
        return (
            max(pharma_match_score_list),
            sascorer.calculateScore(mol) * 0.1,
            QED.default(mol),
            loaded_reg.predict([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]).item()
        )


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class CSDDataset(InMemoryDataset):

    def __init__(self, stage, root, remove_h:bool, prop2idx, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        self.prop2idx = prop2idx
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
        

    @property
    def raw_file_names(self):
        return ['CSD_process.sdf', 'CSD_prop.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']
    def download(self):
        # 添加对药效团匹配评分以及SA， QED的计算
        # first select the best pharmacophore model
        # target_path = '/raid/yyw/PharmDiGress/data/PDK1_pdb'
        # files = os.listdir(target_path)
        # target_mols = []
        # for file in files:
        #     if file.endswith(".sdf"):
        #         target_mol = Chem.SDMolSupplier(os.path.join(target_path,file), removeHs=False, sanitize=True)
        #         target_mols.append(target_mol[0])

        # selected_mol, pp_graph_list = get_best_match_phar_models(list(filter(None, target_mols)))
        # writer = Chem.SDWriter(os.path.join(target_path, 'choose_mols.sdf'))
        # for mol in selected_mol:
        #     writer.write(mol)
        # writer.close()
        # 导入药效团以及毒性预测模型
        target = list(self.prop2idx.keys())[0].split('_')[0]
        pp_graph_list, _ = load_graphs(f"/raid/yyw/PharmDiGress/data/{target}_pdb/{target}_phar_graphs.bin")
        for pp_graph in pp_graph_list:
            pp_graph.ndata['h'] = \
                torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
        loaded_reg = joblib.load('/raid/yyw/PharmDiGress/data/stacking_regressor_model_1.pkl')
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)  
        import csv
        # 创建进程池
        with multiprocessing.Pool(8) as pool:
            with open(self.raw_paths[1], 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([f"{target}","SA","QED", "acute_tox"])
            # 使用 tqdm 迭代并监视分子列表
                for result in pool.imap(process_mol, tqdm([(mol, pp_graph_list, loaded_reg) for mol in suppl], total=len(suppl))):
                    writer.writerow(result)
                writer.close()
        pool.close()
        df = pd.read_csv(self.raw_paths[1])
        rng = RandomState()
        train = df.sample(frac=0.6, random_state=rng)
        val = df.loc[~df.index.isin(train.index)].sample(frac=0.5, random_state=rng)
        test = df.drop(train.index.union(val.index))

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {'H': 0,'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Mg': 6, 'P': 7, 
                'S': 8, 'Cl': 9, 'Ca': 10, 'Br': 11, 'I': 12, 'Ba': 13}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        charge_dict = {'H': 1,'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Mg': 12, 'P': 15, 
                        'S': 16, 'Cl': 17, 'Ca': 20, 'Br': 35, 'I': 53, 'Ba': 56}
        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
        f.close()
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=True, sanitize=True)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i not in target_df.index or mol is None:
                continue
            mol = Chem.AddHs(mol)
            N = mol.GetNumAtoms()
            if N > 40:
                continue

            check = True
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            posc = pos - pos.mean(dim=0)

            charges = []
            formal_charges = []

            type_idx = []
            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                type_idx.append(types[atom_str])
                charges.append(charge_dict[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                if bond.GetBondType() == Chem.rdchem.BondType.UNSPECIFIED:
                    check = False
                    break
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if check == False:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            # edge_type = edge_type[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            atom_type = torch.tensor(type_idx)
            charges = torch.tensor(charges)
            # y = torch.zeros((1, 0), dtype=torch.float)
            y = target[i].unsqueeze(0)

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                pos = pos[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]
                atom_type = atom_type[to_keep] 
                charges = charges[to_keep]

            data = Data(x=x, atom_type=atom_type, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, idx=i, pos=posc, charge=charges, fc=torch.tensor(formal_charges),
                        rdmol=copy.deepcopy(mol))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    def compute_property_mean_mad(self, prop2idx):
        prop_values = []

        prop_ids = torch.tensor(list(prop2idx.values()))
        for idx in range(len(self.indices())):
            data = self.get(self.indices()[idx])
            tars = []
            for prop_id in prop_ids:
                tars.append(data.y[0][prop_id].reshape(1))
            tars = torch.cat(tars)
            prop_values.append(tars)
        prop_values = torch.stack(prop_values, dim=0)
        mean = torch.mean(prop_values, dim=0, keepdim=True)
        ma = torch.abs(prop_values - mean)
        mad = torch.mean(ma, dim=0)

        prop_norm = {}
        for tmp_i, key in enumerate(prop2idx.keys()):
            prop_norm[key] = {
                'mean': mean[0, tmp_i].item(),
                'mad': mad[tmp_i].item()
            }
        return prop_norm


class CSDDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        target = getattr(cfg.general, 'guidance_target')
        regressor = getattr(cfg.general, 'regressor')
        prop2idx =  {'glp1_score' :0, 'cav32_score':1, 'hpk1_score' :2, 'lrrk2_score' :3, 'pharma_score' :4, 'SA' :5, 'QED':6, 'acute_tox':7}
        if self.remove_h:
            atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Mg': 5, 'P': 6, 
                            'S': 7, 'Cl': 8, 'Ca': 9, 'Br': 10, 'I': 11, 'Ba': 12}
            atom_index = {5: 1, 6: 2, 7: 3, 8: 4, 9:5, 12:6, 15:7, 16:8, 17:9, 20:10, 
                            35:11, 53:12, 56:13}
        else:
            atom_encoder = {'H': 0,'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Mg': 6, 'P': 7, 
                            'S': 8, 'Cl': 9, 'Ca': 10, 'Br': 11, 'I': 12, 'Ba': 13}
            atom_index = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9:5, 12:6, 15:7, 16:8, 17:9, 20:10, 
                            35:11, 53:12, 56:13}
        if regressor and target == 'EdgeComCond':

            transform = EdgeComCondTransform(atom_encoder.values(), atom_index, include_aromatic=True,
                                             property_idx=prop2idx[cfg.model.context])
        elif regressor and target == 'EdgeComCondMulti':

            transform = EdgeComCondMultiTransform(atom_encoder.values(), atom_index, include_aromatic=True,
                                                  property_idx1=prop2idx[cfg.model.context[0]], property_idx2=prop2idx[cfg.model.context[1]],
                                                  property_idx3=prop2idx[cfg.model.context[2]], property_idx4=prop2idx[cfg.model.context[3]])
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': CSDDataset(stage='train', root=root_path,remove_h = self.remove_h, prop2idx=prop2idx,
                                        transform=transform),
                    'val': CSDDataset(stage='val', root=root_path, remove_h = self.remove_h, prop2idx=prop2idx,
                                      transform=transform),
                    'test': CSDDataset(stage='test', root=root_path, remove_h = self.remove_h, prop2idx=prop2idx,
                                       transform=transform)}
        super().__init__(cfg, datasets)


class CSDinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.remove_h = cfg.dataset.remove_h
        self.name = 'csd'
        if self.remove_h:
            self.atom_index = {5: 0, 6: 1, 7: 2, 8: 3, 9:4, 12:5, 15:6, 16:7, 17:8, 20:9, 
                            35:10, 53:11, 56:12}
            self.atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Mg': 5, 'P':6, 
                                'S': 7, 'Cl': 8, 'Ca': 9, 'Br': 10, 'I': 11, 'Ba': 12}
            self.atom_decoder = ['B', 'C', 'N', 'O', 'F', 'Mg', 'P', 
                                'S', 'Cl', 'Ca', 'Br', 'I', 'Ba']
            self.valencies = [3, 4, 3, 2, 1, 2, 3, 2, 1, 2, 1, 1, 2]
            self.num_atom_types = len(self.atom_decoder)
            self.max_n_nodes = 40
            self.max_weight = 600
            self.atom_weights = {0: 11, 1: 12, 2: 14, 3: 16, 4:19, 5:24, 6:31, 7:32, 
                                8:35, 9:40, 10:80, 11:127, 12:137}
            self.prop2idx = {value: index for index, value in enumerate(getattr(cfg.model, 'context'))}
            self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 3.7631e-04, 9.4830e-04, 2.0170e-03, 2.8148e-03,
                                        5.1479e-03, 5.5543e-03, 1.2779e-02, 1.4872e-02, 2.7832e-02, 2.9141e-02,
                                        4.2342e-02, 4.0777e-02, 5.9803e-02, 6.0856e-02, 8.3495e-02, 8.3510e-02,
                                        1.0476e-01, 9.1277e-02, 9.3987e-02, 7.1438e-02, 6.6034e-02, 3.7616e-02,
                                        3.0812e-02, 1.3171e-02, 9.1368e-03, 3.0707e-03, 2.6342e-03, 7.6767e-04,
                                        1.1139e-03, 2.5589e-04, 5.4188e-04, 1.9568e-04, 4.0641e-04, 1.0537e-04,
                                        2.1073e-04, 6.0209e-05, 3.0105e-05, 6.0209e-05, 4.5157e-05])
            self.node_types = torch.tensor([6.1698e-04, 7.3823e-01, 7.2801e-02, 1.2315e-01, 1.7945e-02, 1.6145e-05,
                                            2.5314e-03, 2.0910e-02, 1.3963e-02, 3.4597e-06, 7.8282e-03, 2.0066e-03,
                                            0.0000e+00])
            self.edge_types = torch.tensor([0.8771, 0.0590, 0.0109, 0.0009, 0.0521])
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:9] = torch.tensor([0.0000e+00, 1.3238e-01, 2.0985e-01, 3.8483e-01, 2.6112e-01, 7.4222e-03,
                                                        4.3973e-03, 0.0000e+00, 2.3065e-06])
        else:
            self.atom_index = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9:5, 12:6, 15:7, 16:8, 17:9, 20:10, 
                            35:11, 53:12, 56:13}
            self.atom_encoder = {'H': 0,'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Mg': 6, 'P': 7, 
                                'S': 8, 'Cl': 9, 'Ca': 10, 'Br': 11, 'I': 12, 'Ba': 13}
            self.atom_decoder = ['H','B', 'C', 'N', 'O', 'F', 'Mg', 'P', 
                                'S', 'Cl', 'Ca', 'Br', 'I', 'Ba']
            self.valencies = [1, 3, 4, 3, 2, 1, 2, 3, 2, 1, 2, 1, 1, 2]
            self.num_atom_types = len(self.atom_decoder)
            self.max_n_nodes = 40
            self.max_weight = 500
            self.atom_weights = {0: 1, 1: 11, 2: 12, 3: 14, 4: 16, 5:19, 6:24, 7:31, 8:32, 
                                9:35, 10:40, 11:80, 12:127, 13:137}
            self.prop2idx = {value: index for index, value in enumerate(getattr(cfg.model, 'context'))}
            self.n_nodes = torch.tensor([0.0000, 0.0000, 0.0001, 0.0004, 0.0009, 0.0013, 0.0023, 0.0026, 0.0059,
                                        0.0072, 0.0133, 0.0140, 0.0204, 0.0195, 0.0287, 0.0292, 0.0407, 0.0420,
                                        0.0546, 0.0504, 0.0572, 0.0507, 0.0582, 0.0496, 0.0553, 0.0457, 0.0478,
                                        0.0367, 0.0398, 0.0297, 0.0341, 0.0234, 0.0270, 0.0186, 0.0221, 0.0138,
                                        0.0174, 0.0102, 0.0121, 0.0073, 0.0099])
            self.node_types = torch.tensor([3.8415e-04, 8.3435e-04, 7.8233e-01, 5.6424e-02, 1.1735e-01, 1.1164e-02,
                                            9.6143e-06, 3.0716e-03, 1.4705e-02, 7.9167e-03, 2.9261e-06, 4.7783e-03,
                                            1.0296e-03, 0.0000e+00])
            self.edge_types = torch.tensor([9.1021e-01, 6.5449e-02, 2.3911e-02, 4.3119e-04, 0.0000e+00])
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:9] = torch.tensor([0.0000e+00, 1.2907e-01, 2.3079e-01, 3.8297e-01, 2.5078e-01, 2.0696e-03,
                                                        4.3273e-03, 4.1801e-07, 8.3602e-07])

        if recompute_statistics or self.n_nodes is None:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts(self.max_n_nodes + 1)
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())
        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_csd_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            pos, pos_mask = to_dense_batch(x=data.pos, batch=data.batch)
            pos = pos * pos_mask.unsqueeze(-1)

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                conformers = pos[k, :n, :3].cpu()
                all_molecules.append([atom_types, edge_types, conformers, data.rdmol])
                # all_molecules.append([atom_types, edge_types])
        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_csd_smiles(atom_decoder, train_dataloader, remove_h):
    '''
    :return:
    '''
    print(f"\tConverting CSD dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X,dense_data.E
        pos, pos_mask = to_dense_batch(x=data.pos, batch=data.batch)
        pos = pos * pos_mask.unsqueeze(-1)
        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            conformers = pos[k, :n, :3].cpu()
            molecule_list.append([atom_types, edge_types, conformers, data.rdmol])
            # molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], molecule[2], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting CSD dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

