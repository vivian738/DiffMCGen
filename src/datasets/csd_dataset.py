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

    def __init__(self, stage, root, prop2idx, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
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

        return ['proc_tr.pt', 'proc_val.pt', 'proc_test.pt']
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
        loaded_reg = joblib.load('/raid/yyw/PharmDiGress/data/stacking_regressor_model.pkl')
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
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i not in target_df.index or mol is None:
                continue

            N = mol.GetNumAtoms()
            if N > 40:
                continue

            check = True
            if mol.GetConformer() is None:
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
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
            # y = torch.zeros((1, 0), dtype=torch.float)
            y = target[i].unsqueeze(0)

            data = Data(x=x, atom_type=atom_type, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, idx=i, pos=pos, charge=torch.tensor(charges), fc=torch.tensor(formal_charges),
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
        self.remove_h = False
        target = getattr(cfg.general, 'guidance_target')
        regressor = getattr(cfg.general, 'regressor')
        prop2idx =  {'glp1_score' :0, 'cav32_score':1, 'hpk1_score' :2, 'lrrk2_score' :3, 'pharma_score' :4, 'SA' :5, 'QED':6, 'acute_tox':7}

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
        datasets = {'train': CSDDataset(stage='train', root=root_path,prop2idx=prop2idx,
                                        transform=transform),
                    'val': CSDDataset(stage='val', root=root_path, prop2idx=prop2idx,
                                      transform=transform),
                    'test': CSDDataset(stage='test', root=root_path,prop2idx=prop2idx,
                                       transform=transform)}
        super().__init__(cfg, datasets)


class CSDinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.remove_h = False
        self.name = 'csd'

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
        self.n_nodes = torch.tensor([0.0000, 0.0000, 0.0001, 0.0004, 0.0010, 0.0013, 0.0023, 0.0027, 0.0057,
                                    0.0072, 0.0134, 0.0137, 0.0200, 0.0193, 0.0288, 0.0292, 0.0408, 0.0420,
                                    0.0543, 0.0506, 0.0576, 0.0507, 0.0585, 0.0492, 0.0554, 0.0459, 0.0481,
                                    0.0372, 0.0396, 0.0293, 0.0338, 0.0235, 0.0273, 0.0186, 0.0215, 0.0137,
                                    0.0174, 0.0100, 0.0122, 0.0073, 0.0101])
        self.node_types = torch.tensor([3.3095e-04, 9.1393e-04, 7.8595e-01, 5.5002e-02, 1.1528e-01, 1.2172e-02,
                                        1.4454e-05, 3.2918e-03, 1.4168e-02, 7.4630e-03, 4.4473e-06, 4.4840e-03,
                                        9.3394e-04, 0.0000e+00])
        self.edge_types = torch.tensor([9.1977e-01, 5.8504e-02, 2.1344e-02, 3.8092e-04, 0.0000e+00])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:9] = torch.tensor([0.0000e+00, 1.2928e-01, 2.2961e-01, 3.8086e-01, 2.5391e-01, 2.1662e-03,
                                                    4.1731e-03, 1.1118e-06, 1.1118e-06])

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
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_csd_smiles(atom_decoder, train_dataloader)
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


def compute_csd_smiles(atom_decoder, train_dataloader):
    '''
    :return:
    '''
    print(f"\tConverting CSD dataset to SMILES ...")

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

