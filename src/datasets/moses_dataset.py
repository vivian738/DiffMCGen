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
from datasets.qm9_dataset import EdgeComCondTransform, EdgeComCondMultiTransform
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


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    def __init__(self, stage, root, filter_dataset: bool, remove_h: bool, point_phore: list, transform=None, pre_transform=None, pre_filter=None):
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
        pp_graph_list, _ = load_graphs("/raid/yyw/PharmDiGress/data/PDK1_pdb/pdk1_phar_graphs.bin")
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
        smiles_list = pd.read_csv(path)['smiles'].values
        phar_score = pd.read_csv(path)['pharma_score'].values
        sa = pd.read_csv(path)['SA'].values
        qed = pd.read_csv(path)['QED'].values
        acute_tox = pd.read_csv(path)['acute_tox'].values
        with open(path, 'r') as f:
            tmp = list(zip(phar_score, sa, qed, acute_tox))
            target = [list(v) for v in tmp]
            target = torch.tensor(target, dtype=torch.float)

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
        prop2idx = {'pharma_score': 0, 'SA': 1, 'QED': 2, 'acute_tox': 3}

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
                                          remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, transform=transform),
                    'val': MOSESDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset, 
                                        remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, transform=transform),
                    'test': MOSESDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset, 
                                        remove_h=cfg.dataset.remove_h, point_phore=cfg.dataset.phore, transform=transform)}
        super().__init__(cfg, datasets)



class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'MOSES'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = cfg.dataset.remove_h
        self.prop2idx = {'pharma_score': 0, 'SA': 1, 'QED': 2, 'acute_tox_score': 3}


        if self.remove_h == False:
            self.atom_decoder = ['H', 'C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 32, 4: 16, 5: 19, 6: 35.4, 7: 79.9}
            self.valencies = [1, 4, 3, 4, 2, 1, 1, 1]
            self.num_atom_types = len(self.atom_decoder)
            self.max_weight = 700
            
            meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                            node_types=f'{self.name}_atom_types.txt',
                            edge_types=f'{self.name}_edge_types.txt',
                            valency_distribution=f'{self.name}_valencies.txt')

            self.n_nodes = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.097634362347889692e-06,
                                        1.858580617408733815e-05, 5.007842264603823423e-05, 5.678996240021660924e-05,
                                        1.244216400664299726e-04, 4.486406978685408831e-04, 2.253012731671333313e-03,
                                        3.231865121051669121e-03, 6.709992419928312302e-03, 2.289564721286296844e-02,
                                        5.411050841212272644e-02, 1.099515631794929504e-01, 1.223291903734207153e-01,
                                        1.280680745840072632e-01, 1.445975750684738159e-01, 1.505961418151855469e-01,
                                        1.436946094036102295e-01, 9.265746921300888062e-02, 1.820066757500171661e-02,
                                        2.065089574898593128e-06])
            self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
            self.node_types = torch.tensor([0.0, 0.722338, 0.13661, 0.163655, 0.103549, 0.1421803, 0.005411, 0.00150])
            self.edge_types = torch.tensor([0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486])
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:7] = torch.tensor([0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485])
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

            self.n_nodes = None
            self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
            self.node_types = None
            self.edge_types = None
            self.valency_distribution = None
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