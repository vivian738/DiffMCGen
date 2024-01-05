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
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

from src.eval import sascorer
from rdkit.Chem import QED
from src.datasets.pharmacophore_eval import mol2ppgraph, match_score

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.,
    1., 1., 1.
])

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
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

            data.y = properties[0, self.property_idx:self.property_idx+1] - \
                            torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
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
            if prop_idx == 11:
                Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
                atom_types = data.atom_type
                atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

                property_data.append(properties[0, prop_idx:prop_idx+1] - \
                                torch.sum((atom_counts * torch.tensor(Cv_atomref))))
            else:
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
        data.charge = None
        atom_type = data.atom_type
        one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.one_hot = one_hot.float()
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
            data.y = data.y[0, 11] - torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
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


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class QM9Dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, stage, root, remove_h: bool, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.target_prop = target_prop
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

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
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])
        # 添加对药效团匹配评分以及SA， QED的计算
        target_path = '/raid/yyw/PharmDiGress/data/target_mol.sdf'
        target_mols = Chem.SDMolSupplier(target_path, removeHs=True, sanitize=True)
        pp_graph_list = [mol2ppgraph(target_mol) for target_mol in target_mols]
        for pp_graph in pp_graph_list:
            pp_graph.ndata['h'] = \
                torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()

        pharma_score = []
        sa = []
        qed = []
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)
        for i, mol in enumerate(tqdm(suppl)):
            if mol is None or '.' in Chem.MolToSmiles(mol):
                pharma_score.append(-1)
                sa.append(-1)
                qed.append(-1)
                continue

            pharma_match_score_list = [match_score(mol, pp_graph) for pp_graph in pp_graph_list]

            pharma_score.append(max(pharma_match_score_list))
            sa.append(sascorer.calculateScore(mol))
            qed.append(QED.default(mol))

        # data = pd.read_csv(self.raw_paths[1])
        dataset['pharma_score'] = pharma_score
        dataset['SA'] = sa
        dataset['QED'] = qed
        dataset.to_csv(self.raw_paths[1], index=False, mode='w')

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'P': 15, 'Cl': 17, 'Br': 35, 'I': 53}

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)

        with open('/raid/yyw/PharmDiGress/data/qm9/precess_qdb9.csv', 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[2:24]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            N = mol.GetNumAtoms()
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

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

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
                if prop_id == 11:
                    tars.append(self.sub_Cv_thermo(data).reshape(1))
                else:
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


class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h

        target = getattr(cfg.general, 'guidance_target')
        regressor = getattr(cfg.general, 'regressor')
        prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11, 'pharma_score': 16, 'SA': 17,
                    'QED': 18}
        if self.remove_h:
            atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'P': 5, 'Cl': 6, 'Br': 7, 'I': 8}
            atom_index = {6: 0, 7: 1, 8: 2, 9: 3}
        else:
            atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'P': 6, 'Cl': 7, 'Br': 8, 'I': 9}
            atom_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        if regressor and target == 'EdgeComCond':

            transform = EdgeComCondTransform(atom_encoder.values(), atom_index, include_aromatic=True,
                                             property_idx=prop2idx[cfg.model.context])
        elif regressor and target == 'EdgeComCondMulti':

            transform = EdgeComCondMultiTransform(atom_encoder.values(), atom_index, include_aromatic=True,
                                                  property_idx1=prop2idx[cfg.model.context[0]], property_idx2=prop2idx[cfg.model.context[1]],
                                                  property_idx3=prop2idx[cfg.model.context[2]], property_idx4=prop2idx[cfg.model.context[3]])
        elif regressor and target == 'mu':
            transform = SelectMuTransform()
        elif regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=cfg.dataset.remove_h,
                                        target_prop=target, transform=transform),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=cfg.dataset.remove_h,
                                      target_prop=target, transform=transform),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().__init__(cfg, datasets)


class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        if self.remove_h:
            self.atom_index = {6: 0, 7: 1, 8: 2, 9: 3}
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 9
            self.max_weight = 150
            self.n_nodes = torch.tensor([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
                                         0.0046472, 0.023985, 0.13666, 0.83337])
            self.node_types = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
            self.edge_types = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])
            self.prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11, 'pharma_score': 16, 'SA': 17,
                             'QED': 18}
        else:
            self.atom_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                         9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                         1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                         1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                         5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

            self.node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            self.edge_types = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])
            self.prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11, 'pharma_score': 16,
                             'SA': 17, 'QED': 18}

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            assert False


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
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask, edge_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask, edge_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
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
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

