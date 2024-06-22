from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import copy
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

import utils as utils
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule

from eval import sascorer
from rdkit.Chem import QED, AllChem
from datasets.pharmacophore_eval import mol2ppgraph, match_score, get_best_match_phar_models
from dgl.data.utils import load_graphs
import multiprocessing

import joblib
import csv

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data

def process_mol(args):
    smiles, pp_graph_list, loaded_reg = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles, 0, 0, 0, 0]  # 如果分子无效，返回默认值
    else:
        pharma_match_score_list = [match_score(mol, pp_graph) for pp_graph in pp_graph_list]
        return [
            smiles,
            max(pharma_match_score_list),
            sascorer.calculateScore(mol) * 0.1,
            QED.default(mol),
            loaded_reg.predict([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]).item()
        ]

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
        properties = data.y
        prop_list = [self.property_idx1, self.property_idx2, self.property_idx3, self.property_idx4]
        property_data = []
        for prop_idx in prop_list:
            property = properties[0, prop_idx:prop_idx+1]
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

class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    def __init__(self, stage, root, filter_dataset: bool, remove_h: bool, point_phore: list, prop2idx:dict, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.remove_h = remove_h
        self.filter_dataset = filter_dataset
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.phore = point_phore
        self.prop2idx = prop2idx
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
        

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_file_name(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'test_filtered.pt', 'test_scaffold_filtered.pt']
        else:
            return ['train.pt', 'test.pt', 'test_scaffold.pt']

    def download(self):

        if files_exist(self.split_paths):
            return
        
        train_path = download_url(self.train_url, self.raw_dir)

        test_path = download_url(self.test_url, self.raw_dir)

        valid_path = download_url(self.val_url, self.raw_dir)

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
        target = list(self.prop2idx.keys())[0].split('_')[0]
        pp_graph_list, _ = load_graphs("/raid/yyw/PharmDiGress/data/{target}_pdb/{target}_phar_graphs.bin")
        for pp_graph in pp_graph_list:
            pp_graph.ndata['h'] = \
                torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
        loaded_reg = joblib.load('/raid/yyw/PharmDiGress/data/stacking_regressor_model.pkl')

        path = [train_path, test_path, valid_path]

        for n,i in enumerate(path):
            dataset=pd.read_csv(i)
            if n == 0:
                sub_path = osp.join(self.raw_dir, 'train_moses.csv')
            elif n == 1:
                sub_path = osp.join(self.raw_dir, 'test_moses.csv')
            else:
                sub_path = osp.join(self.raw_dir, 'val_moses.csv')

            with open(sub_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["smiles","pharma_score","SA","QED", "acute_tox"])
                with multiprocessing.Pool(10) as pool:
                    # 使用 tqdm 迭代并监视分子列表
                    for result in tqdm(pool.imap_unordered(process_mol, [(smiles, pp_graph_list, loaded_reg) for smiles in dataset['SMILES'].values]), total=len(dataset['SMILES'].values)):
                        writer.writerow(result)

                    pool.close()
                writer.close()
        

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        if self.remove_h == False:
            atom_decoder = ['H', 'C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35}
        else:
            atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            charge_dict = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35}

        types = {atom: i for i, atom in enumerate(atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        if self.file_idx == 0:
            sampled_df = pd.read_csv(path).sample(frac=1/3, random_state=42)
        else:
            sampled_df = pd.read_csv(path)
        smiles_list = sampled_df['smiles'].values
        sampled_array = sampled_df.iloc[:, 1:].to_numpy()
        target = torch.tensor(sampled_array, dtype=torch.float)

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()
            m = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(m, useRandomCoords=True)
                AllChem.MMFFOptimizeMolecule(m)
            except ValueError:
                continue
                
            mol = Chem.RemoveHs(m)
            # mol = m
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

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
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            atom_type = torch.tensor(type_idx)
            # y = torch.zeros(size=(1, 0), dtype=torch.float)
            y = target[i].unsqueeze(0)

            data = Data(x=x, atom_type=atom_type, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, idx=i, pos=pos, charge=torch.tensor(charges), fc=torch.tensor(formal_charges),
                        rdmol=copy.deepcopy(mol))

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                conformers = data.pos[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, conformers, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

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


class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.phore = cfg.dataset.phore

        target = getattr(cfg.general, 'guidance_target')
        regressor = getattr(cfg.general, 'regressor')
        prop2idx = {'pharma_score':0,'SA':1,'QED':2,'acute_tox':3,'glp1_score':4,'cav32_score':5,'hpk1_score':6,'lrrk2_score':7}

        if cfg.dataset.remove_h == False:
            self.atom_decoder = ['H', 'C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            atom_index = {1: 0, 6: 1, 7: 2, 16: 3, 8: 4, 9: 5, 17: 6, 35: 7}
        else:
            self.atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            atom_index = {6: 1, 7: 2, 16: 3, 8: 4, 9: 5, 17: 6, 35: 7}
        if regressor and target == 'EdgeComCond':

            transform = EdgeComCondTransform(self.atom_encoder.values(), atom_index, include_aromatic=True,
                                             property_idx=prop2idx[cfg.model.context])
        elif regressor and target == 'EdgeComCondMulti':

            transform = EdgeComCondMultiTransform(self.atom_encoder.values(), atom_index, include_aromatic=True,
                                                  property_idx1=prop2idx[cfg.model.context[0]],
                                                  property_idx2=prop2idx[cfg.model.context[1]],
                                                  property_idx3=prop2idx[cfg.model.context[2]],
                                                  property_idx4=prop2idx[cfg.model.context[3]])
        else:
            transform = RemoveYTransform()

        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': MOSESDataset(stage='train', root=root_path, filter_dataset=self.filter_dataset,
                                          remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, prop2idx=prop2idx, transform=transform),
                    'val': MOSESDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset, 
                                        remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, prop2idx=prop2idx, transform=transform),
                    'test': MOSESDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset, 
                                        remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, prop2idx=prop2idx, transform=transform)}
        super().__init__(cfg, datasets)



class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'MOSES'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = cfg.dataset.remove_h
        self.prop2idx = {value: index for index, value in enumerate(getattr(cfg.model, 'context'))}


        if self.remove_h == False:
            self.atom_decoder = ['H', 'C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 32, 4: 16, 5: 19, 6: 35.4, 7: 79.9}
            self.valencies = [1, 4, 3, 4, 2, 1, 1, 1]
            self.num_atom_types = len(self.atom_decoder)
            self.max_weight = 500
            
            meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                            node_types=f'{self.name}_atom_types.txt',
                            edge_types=f'{self.name}_edge_types.txt',
                            valency_distribution=f'{self.name}_valencies.txt')

            self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                        0.0000e+00, 0.0000e+00, 2.8397e-06, 0.0000e+00, 5.6795e-07, 5.6795e-06,
                                        3.9756e-06, 1.8742e-05, 3.1805e-05, 3.1805e-05, 6.3610e-05, 1.1359e-04,
                                        1.6584e-04, 1.7493e-04, 3.5099e-04, 4.7480e-04, 8.1046e-04, 1.2824e-03,
                                        1.8782e-03, 2.7392e-03, 4.4584e-03, 6.3536e-03, 8.7890e-03, 1.2624e-02,
                                        1.7098e-02, 2.2378e-02, 2.9360e-02, 3.5387e-02, 4.2517e-02, 4.9722e-02,
                                        5.5761e-02, 6.0463e-02, 6.5334e-02, 6.6398e-02, 6.8688e-02, 6.6917e-02,
                                        6.3352e-02, 5.9485e-02, 5.3930e-02, 4.7901e-02, 3.9722e-02, 3.2245e-02,
                                        2.4240e-02, 2.0185e-02, 1.3103e-02, 9.1860e-03, 6.1918e-03, 4.4323e-03,
                                        2.7244e-03, 1.3830e-03, 9.4563e-04, 3.3850e-04, 1.4823e-04, 5.3955e-05,
                                        2.7829e-05, 6.2474e-06, 1.1359e-06])
            self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
            self.node_types = torch.tensor([0.4538, 0.3943, 0.0745, 0.0089, 0.0567, 0.0078, 0.0030, 0.0008])
            self.edge_types = torch.tensor([9.4731e-01, 3.7415e-02, 2.0810e-03, 1.0259e-04, 1.3092e-02])
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:8] = torch.tensor([0.0000e+00, 4.0108e-01, 8.2538e-02, 1.1124e-01, 4.0108e-01, 1.1953e-06,
        4.0429e-03, 2.0918e-05])
        else:
            self.atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            self.atom_weights = {1: 12, 2: 14, 3: 32, 4: 16, 5: 19, 6: 35.4, 7: 79.9}
            self.valencies = [4, 3, 4, 2, 1, 1, 1]
            self.num_atom_types = len(self.atom_decoder)
            self.max_weight = 500
            
            meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                            node_types=f'{self.name}_atom_types.txt',
                            edge_types=f'{self.name}_edge_types.txt',
                            valency_distribution=f'{self.name}_valencies.txt')

            self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                        0.0000e+00, 0.0000e+00, 3.4077e-06, 2.0446e-05, 5.5091e-05, 6.0202e-05,
                                        1.3517e-04, 4.6345e-04, 2.3524e-03, 3.3492e-03, 6.8932e-03, 2.3576e-02,
                                        5.5278e-02, 1.1133e-01, 1.2298e-01, 1.2805e-01, 1.4400e-01, 1.4964e-01,
                                        1.4232e-01, 9.1484e-02, 1.8010e-02, 1.7038e-06])
            self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
            self.node_types = torch.tensor([0.7220, 0.1364, 0.0164, 0.1038, 0.0143, 0.0055, 0.0015])
            self.edge_types = torch.tensor([8.9733e-01, 4.7399e-02, 6.3060e-03, 3.5479e-04, 4.8611e-02])
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:7] = torch.tensor([0.0000, 0.1063, 0.2730, 0.3604, 0.2498, 0.0054, 0.0050])
        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())

        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            pos, pos_mask = to_dense_batch(x=data.pos, batch=data.batch)
            pos = pos * pos_mask.unsqueeze(-1)

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                positions = pos[k, :n, :3].cpu()
                all_molecules.append([atom_types, edge_types, positions, data.rdmol])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


if __name__ == "__main__":
    ds = [MOSESDataset(s, os.path.join(os.path.abspath(__file__), "../../../data/moses"),
                       preprocess=True) for s in ["train", "val", "test"]]