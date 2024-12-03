import math
import torch
from torch_scatter import scatter_add, scatter_mean
from rdkit import Chem
# from rdkit.Chem import rdMolTransforms


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.type_as(x)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
    return score_pos


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def is_radius_edge(edge_type):
    return edge_type == 0

def get_angles(mol_list, bidirectional=True):
    atom_counter = 0
    bondList = []
    angleList = []
    for m in mol_list:
        bondSmarts = '*~*~*'
        bondQuery = Chem.MolFromSmarts(bondSmarts)
        matches = m.GetSubstructMatches(bondQuery)
        conf = m.GetConformer()
        for match in matches:
            idx0 = match[0]
            idx1 = match[1]
            idx2 = match[2]
            bondList.append([idx0+atom_counter, idx1+atom_counter, idx2+atom_counter])
            angleList.append(rdMolTransforms.GetAngleRad(conf, idx0, idx1, idx2))
            if bidirectional:
                bondList.append([idx2+atom_counter, idx1+atom_counter, idx0+atom_counter])
                angleList.append(rdMolTransforms.GetAngleRad(conf, idx2, idx1, idx0))
        atom_counter += m.GetNumAtoms()
    return bondList, angleList


def get_torsions(mol_list):
    atom_counter=0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
                or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
                continue
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                        or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    torsionList.append([idx1+atom_counter, idx2+atom_counter, idx3+atom_counter, idx4+atom_counter])
                    dihedralList.append(rdMolTransforms.GetDihedralRad(conf, idx1, idx2, idx3, idx4))
    return torsionList, dihedralList
