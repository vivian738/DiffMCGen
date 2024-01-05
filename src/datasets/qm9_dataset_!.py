import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import subgraph
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
import copy
from functools import lru_cache
from rdkit import Chem
from src.eval import sascorer
from rdkit.Chem import QED
from src.datasets.pharmacophore_eval import mol2ppgraph, match_score
import pathlib
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, MolecularDataModule
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics
import src.utils as utils

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, HAR2EV, HAR2EV, HAR2EV,
    1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root: str,
                 condition = None,
                 transform= None,
                 pre_transform = None,
                 pre_filter = None):

        self.condition = condition
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])


    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']
    @property
    def processed_file_names(self) -> str:
        return ['data_qm9.pt']

    def _download(self):
        if files_exist(self.processed_paths):
            return

        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                  osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            raise ImportError("Please install 'rdkit' to alternatively process the raw data.")

        if self.condition is not None:
            if self.condition == 'gap':
                p_threshold = 4.5 * 0.036749344227551  # 1ev = 0.036749344227551 Ha
            elif self.condition == 'alpha':
                p_threshold = 91
            else:
                raise Exception('Wrong condition name')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'P': 6, 'Cl' : 7, 'Br' : 8, 'I' : 9}
        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
        charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'P': 15, 'Cl' : 17, 'Br' : 35, 'I' : 53}

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=True)
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        processdata_path = '/raid/yyw/DiGress/data/qm9/precess_qdb9.csv'
        if os.path.exists(processdata_path):
            print('condition is already added in target.csv')
        else:
            # 添加对药效团匹配评分以及SA， QED的计算
            target_path = '/raid/yyw/DiGress/data/target_mol.sdf'
            target_mols = Chem.SDMolSupplier(target_path, removeHs=True, sanitize=True)
            pp_graph_list = [mol2ppgraph(target_mol) for target_mol in target_mols]
            for pp_graph in pp_graph_list:
                pp_graph.ndata['h'] = \
                    torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
                pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()

            pharma_score = []
            sa = []
            qed = []
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

            data = pd.read_csv(self.raw_paths[1])
            data['pharma_score'] = pharma_score
            data['SA'] = sa
            data['QED'] = qed
            data.to_csv('/raid/yyw/DiGress/data/qm9/precess_qdb9.csv')

            # 导入标签
        with open('/raid/yyw/DiGress/data/qm9/precess_qdb9.csv', 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[2:24]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or mol is None:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            charges = []
            formal_charges = []

            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                type_idx.append(types[atom_str])
                charges.append(charge_dict[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                if bond.GetBondType() == BT.AROMATIC:
                    print('meet aromatic bond!')

                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            atom_type = torch.tensor(type_idx)
            # y = [torch.tensor(target[name][i], dtype=torch.float32) for name in
            #      ['mu', 'alpha', 'homo', 'lumo',
            #       'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv;',
            #       'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'A', 'B', 'C', ]]


            y = target[i].unsqueeze(0)
            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            # name = mol.GetProp('_Name')

            data = Data(x=x, atom_type=atom_type, pos=pos, charge=torch.tensor(charges), fc=torch.tensor(formal_charges),
                        edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, y=y, num_atom=N, idx=i, rdmol=copy.deepcopy(mol))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])

    # def get_idx_split(self, train_size, valid_size, seed):
    #     # load split idx for train, val, test
    #     split_path = osp.join(self.processed_dir, 'split_dict_qm9.pt')
    #     if osp.exists(split_path):
    #         print('Loading existing split data.')
    #         return torch.load(split_path)
    #
    #     data_size = len(self.indices())
    #     # assert data_num == 130831
    #     # train_num = 100000
    #     # test_num = int(0.1 * data_num)
    #     # valid_num = data_num - (train_num + test_num)
    #
    #     # Generate random permutation
    #     np.random.seed(0)
    #     data_perm = np.random.permutation(data_size)
    #     train, valid, test, extra = np.split(
    #         data_perm, [train_size, train_size + valid_size, train_size + valid_size + test_size])
    #
    #     train = np.array(self.indices())[train]
    #     valid = np.array(self.indices())[valid]
    #     test = np.array(self.indices())[test]
    #
    #     splits = {'train': train, 'valid': valid, 'test': test}
    #     torch.save(splits, split_path)
    #     return splits

    def get_cond_idx_split(self):
        # construct conditional generation split idx for first train, second train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_cond_qm9.pt')
        if osp.exists(split_path):
            print('Loading existing split data.')
            return torch.load(split_path)

        data_size = len(self.indices())
        train_size = int(0.75 * data_size)
        test_size = int(0.1 * data_size)
        valid_size = int(0.15 * data_size)


        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(data_size)
        train, valid, test, extra = np.split(
            data_perm, [train_size, train_size + valid_size, train_size + valid_size + test_size])

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits

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



class QM9Dataset(QM9):
    def download(self):
        super(QM9Dataset, self).download()

    def process(self):
        super(QM9Dataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)
        return data

class QM9infos(AbstractDatasetInfos):
    def __init__(self, train_loader, val_loader, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        if self.remove_h:
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
        else:
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

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            abstractmodule = AbstractDataModule(train_loader=train_loader, val_loader=val_loader, cfg=cfg)
            # self.n_nodes = datamodule.node_counts()
            self.n_nodes = abstractmodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = abstractmodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = abstractmodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = train_loader.valency_count(self.max_n_nodes)
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
            dense_data, node_mask, edge_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
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
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
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