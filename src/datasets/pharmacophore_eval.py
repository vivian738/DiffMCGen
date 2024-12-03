import os
import random

import numpy as np
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import ChemicalFeatures

import torch

RDLogger.DisableLog('rdApp.*')

from itertools import product, permutations
from collections import Counter
from sklearn.cluster import KMeans
import dgl
from rdkit import Chem
from rdkit import DataStructs
from dgl.data.utils import save_graphs

def six_encoding(atom):
    # actually seven
    orgin_phco = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in atom:
        orgin_phco[j] = 1
    return torch.HalfTensor(orgin_phco[1:])


def mol2ppgraph(mol):
    '''
    :param smiles: a molecule
    :return: (pp_graph, mapping)
        pp_graph: DGLGraph, the corresponding **random** pharmacophore graph
        mapping: np.Array ((atom_num, MAX_NUM_PP_GRAPHS)) the mapping between atoms and pharmacophore features
    '''

    atom_index_list = []
    pharmocophore_all = []

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    for f in feats:
        phar = f.GetFamily()
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        # atom_type = f.GetType()
        mapping = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3,
                   'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
        phar_index = mapping.setdefault(phar, 7)
        pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
        pharmocophore_all.append(pharmocophore_)  # all pharmacophore features within a molecule
        atom_index_list.append(atom_index)  # atom indices of one pharmacophore feature
    random.shuffle(pharmocophore_all)
    num = [3, 4, 5, 6, 7]
    p_all = [sublist[0] for sublist in pharmocophore_all if sublist[0] in num]
    num_p = [p_all.count(i)/len(p_all) for i in num]  # P(Number of Pharmacophore points)
    num_ = sample_probability(num, num_p, 1)

    type_list = []
    size_ = []

    ## The randomly generated clusters are obtained,
    # and the next step is to perform a preliminary merging of these randomly generated clusters with identical elements
    if len(pharmocophore_all) >= int(num_[0]):
        mol_phco = pharmocophore_all[:int(num_[0])]
    else:
        mol_phco = pharmocophore_all

    for pharmocophore_all_i in range(len(mol_phco)):
        for pharmocophore_all_j in range(len(mol_phco)):
            if mol_phco[pharmocophore_all_i][1] == mol_phco[pharmocophore_all_j][1] \
                    and mol_phco[pharmocophore_all_i][0] != mol_phco[pharmocophore_all_j][0]:
                index_ = [min(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0]),
                          max(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0])]
                mol_phco[pharmocophore_all_j] = [index_, mol_phco[pharmocophore_all_i][1]]
                mol_phco[pharmocophore_all_i] = [index_, mol_phco[pharmocophore_all_i][1]]
            else:
                index_ = mol_phco[pharmocophore_all_i][0]
    unique_index_filter = []
    unique_index = []
    for mol_phco_candidate_single in mol_phco:
        if mol_phco_candidate_single not in unique_index:
            if type(mol_phco[0]) == list:
                unique_index.append(mol_phco_candidate_single)
            else:
                unique_index.append([[mol_phco_candidate_single[0]], mol_phco_candidate_single[1]])
    for unique_index_single in unique_index:
        if unique_index_single not in unique_index_filter:
            unique_index_filter.append(unique_index_single)  ## The following is the order of the pharmacophores by atomic number
    sort_index_list = []
    for unique_index_filter_i in unique_index_filter:  ## Collect the mean of the participating elements
        sort_index = sum(unique_index_filter_i[1]) / len(unique_index_filter_i[1])
        sort_index_list.append(sort_index)
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    unique_index_filter_sort = []
    for index_id in sorted_id:
        unique_index_filter_sort.append(unique_index_filter[index_id])
    position_matrix = np.zeros((len(unique_index_filter_sort), len(unique_index_filter_sort)))
    e_list = []
    for mol_phco_i in range(len(unique_index_filter_sort)):
        mol_phco_i_elment = list(unique_index_filter_sort[mol_phco_i][1])
        if type(unique_index_filter_sort[mol_phco_i][0]) == list:
            type_list.append(six_encoding(unique_index_filter_sort[mol_phco_i][0]))
        else:
            type_list.append(six_encoding([unique_index_filter_sort[mol_phco_i][0]]))

        size_.append(len(mol_phco_i_elment))
        e_list.append(mol_phco_i_elment)
        for mol_phco_j in range(len(unique_index_filter_sort)):
            mol_phco_j_elment = list(unique_index_filter_sort[mol_phco_j][1])
            if mol_phco_i_elment == mol_phco_j_elment:
                position_matrix[mol_phco_i, mol_phco_j] = 0
            elif str(set(mol_phco_i_elment).intersection(set(mol_phco_j_elment))) == 'set()':
                dist_set = []
                for atom_i in mol_phco_i_elment:
                    for atom_j in mol_phco_j_elment:
                        dist = cal_dist(mol, atom_i, atom_j)
                        dist_set.append(dist)
                min_dist = min(dist_set)
                if max(len(mol_phco_i_elment), len(mol_phco_j_elment)) == 1:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist
                else:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist + max(len(mol_phco_i_elment),
                                                                             len(mol_phco_j_elment)) * 0.2
            else:
                for type_elment_i in mol_phco_i_elment:
                    for type_elment_j in mol_phco_j_elment:
                        if type_elment_i == type_elment_j:
                            position_matrix[mol_phco_i, mol_phco_j] = max(len(mol_phco_i_elment),
                                                                          len(mol_phco_j_elment)) * 0.2
                        ##The above is a summary of the cases where the two pharmacophores have direct elemental intersection.
    weights = []
    u_list = []
    v_list = []
    phco_single = []

    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            if u != v:
                u_list.append(u)
                v_list.append(v)
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])
    u_list_tensor = torch.tensor(u_list)
    v_list_tensor = torch.tensor(v_list)
    g = dgl.graph((u_list_tensor, v_list_tensor))
    g.edata['dist'] = torch.HalfTensor(weights)
    type_list_tensor = torch.stack(type_list)
    g.ndata['type'] = type_list_tensor
    g.ndata['size'] = torch.HalfTensor(size_)

    return g

def cal_dist(mol, start_atom, end_atom):
    list_ = []
    list_.append(start_atom)
    seen = set()
    seen.add(start_atom)
    parent = {start_atom: None}
    nei_atom = []
    bond_num = mol.GetNumBonds()
    while (len(list_) > 0):
        vertex = (list_[0])
        del (list_[0])
        nei_atom = ([n.GetIdx() for n in mol.GetAtomWithIdx(vertex).GetNeighbors()])   #是QM9数据集中有配位体存在
        for w in nei_atom:
            if w not in seen:
                list_.append(w)
                seen.add(w)
                parent[w] = vertex
    path_atom = []
    if end_atom not in parent.keys():
        parent[end_atom] = parent[start_atom]  ##
    while end_atom != None:
        path_atom.append(end_atom)
        end_atom = parent[end_atom]
        # try:
        #     end_atom = parent[end_atom]
        # except:
        #     end_atom = None
        #     print('end_atom is not in parent')
    nei_bond = []
    for i in range(bond_num):
        nei_bond.append((mol.GetBondWithIdx(i).GetBondType().name, mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                         mol.GetBondWithIdx(i).GetEndAtomIdx()))
    bond_collection = []
    for idx in range(len(path_atom) - 1):
        bond_start = path_atom[idx]
        bond_end = path_atom[idx + 1]
        for bond_type in nei_bond:
            if len(list(set([bond_type[1], bond_type[2]]).intersection(set([bond_start, bond_end])))) == 2:
                bond_ = bond_type[0]
                if [bond_, bond_type[1], bond_type[2]] not in bond_collection:
                    bond_collection.append([bond_, bond_type[1], bond_type[2]])
    dist = 0
    for elment in bond_collection:
        if elment[0] == 'SINGLE':
            dist = dist + 1
        elif elment[0] == 'DOUBLE':
            dist = dist + 0.87
        elif elment[0] == 'AROMATIC':
            dist = dist + 0.91
        else:
            dist = dist + 0.78
    return dist


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = random.randint(0, n - 1)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])
    cresult = Counter(Psample)
    psam = [cresult[x] for x in plist]
    pe = [x * N for x in plist]
    return Psample


def cal_dist_all(mol, phco_list_i, phco_list_j):
    for phco_elment_i in phco_list_i:
        for phco_elment_j in phco_list_j:
            if phco_elment_i == phco_elment_j:
                if len(phco_list_i) == 1 and len(phco_list_j) == 1:
                    dist = 0
                else:
                    dist = max(len(phco_list_i), len(phco_list_j)) * 0.2
        if not set(phco_list_i).intersection(set(phco_list_j)):
            dist_set = []
            for atom_i in phco_list_i:
                for atom_j in phco_list_j:
                    dist_ = cal_dist(mol, atom_i, atom_j)
                    dist_set.append(dist_)
            min_dist = min(dist_set)
            if max(len(phco_list_i), len(phco_list_j)) == 1:
                dist = min_dist
            else:
                dist = min_dist + max(len(phco_list_i), len(phco_list_j)) * 0.2
    return dist


def extract_dgl_info(g):
    node_type = g.ndata.get('type', g.ndata['h'][:, :-1])  # a temporary fix
    dist = g.edata.get('dist', g.edata['h'])

    ref_dist_list = []
    value = []
    for i in range(len(g.edges()[0])):
        ref_dist_name = '{}{}'.format(int(g.edges()[0][i]), int(g.edges()[1][i]))  ##取参考药效团的距离
        ref_dist_list.append(ref_dist_name)
        value.append(float(dist[i]))
    dist_dict = dict(zip(ref_dist_list, value))
    type_list = []
    for n in range(len(node_type)):
        list_0 = [0]
        nonzoro_list = node_type[n].numpy().tolist()
        list_0.extend(nonzoro_list)
        aa = np.nonzero(list_0)
        type_list.append(tuple(aa[0]))
    return dist_dict, type_list


__FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
__MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}


def match_score(mol, g):

    if mol is None:
        return -1
    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    feats = __FACTORY.GetFeaturesForMol(mol)
    dist, ref_type = extract_dgl_info(g)

    all_phar_types = {i for j in ref_type for i in j}

    phar_filter = [[] for _ in range(len(ref_type))]

    phar_mapping = {i: [] for i in ref_type}
    for i in range(len(ref_type)):
        phar_mapping[ref_type[i]].append(i)

    mol_phco_candidate = []
    for f in feats:
        phar = f.GetFamily()
        phar_index = __MAPPING.setdefault(phar, 7)
        if phar_index not in all_phar_types:
            continue
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        phar_info = ((phar_index,), atom_index)
        mol_phco_candidate.append(phar_info)

    tmp_n = len(mol_phco_candidate)
    for i in range(tmp_n):
        phar_i, atom_i = mol_phco_candidate[i]
        for j in range(i + 1, tmp_n):
            phar_j, atom_j = mol_phco_candidate[j]
            if atom_i == atom_j and phar_i != phar_j:
                phars = tuple(sorted((phar_i[0], phar_j[0])))
                mol_phco_candidate.append([phars, atom_i])

    for phar, atoms in mol_phco_candidate:
        if phar in phar_mapping:
            for idx in phar_mapping[phar]:
                phar_filter[idx].append(atoms)

    match_score = max_match(mol, g, phar_filter, phar_mapping.values())
    return match_score


def __iter_product(phco, phar_grouped):
    group_elements = [None for _ in range(len(phar_grouped))]
    n_places = []
    for i in range(len(phar_grouped)):
        group_elements[i] = list(range(len(phco[phar_grouped[i][0]])))
        l_elements = len(group_elements[i])
        l_places = len(phar_grouped[i])
        n_places.append(l_places)

        if l_elements < l_places:
            group_elements[i].extend([None] * (l_places - l_elements))

    for i in product(*[permutations(i, n) for i, n in zip(group_elements, n_places)]):
        res = [None] * len(phco)

        for g_ele, g_idx in zip(i, phar_grouped):
            for a, b in zip(g_ele, g_idx):
                res[b] = a

        yield res


def max_match(mol, g, phco, phar_mapping):
    # will modify phar_filter

    # Chem.SanitizeMol(mol)
    ref_dist, ref_type = extract_dgl_info(g)

    length = len(phco)

    dist_dict = {}
    for i in range(length - 1):
        for j in range(i + 1, length):
            for elment_len1 in range(len(phco[i])):
                for elment_len2 in range(len(phco[j])):
                    if phco[i][elment_len1] is None or phco[j][elment_len2] is None:
                        dist = 100
                    else:
                        dist = cal_dist_all(mol, phco[i][elment_len1], phco[j][elment_len2])  ##

                    dist_name = (i, elment_len1, j, elment_len2)

                    dist_dict[dist_name] = dist

    match_score_max = 0
    for phco_elment_list in __iter_product(phco, list(phar_mapping)):

        error_count = 0
        correct_count = 0

        for p in range(len(phco_elment_list)):
            for q in range(p + 1, len(phco_elment_list)):

                key_ = (p, phco_elment_list[p], q, phco_elment_list[q])

                if phco_elment_list[p] is None or phco_elment_list[q] is None:
                    dist_ref_candidate = 100
                else:
                    dist_ref_candidate = abs(dist_dict[key_] - ref_dist['{}''{}'.format(p, q)])
                if dist_ref_candidate < 1.21:
                    correct_count += 1
                else:
                    error_count += 1
        match_score = correct_count / (correct_count + error_count)

        match_score_max = max(match_score, match_score_max)

        if match_score_max == 1:
            return match_score_max

    return match_score_max

def get_best_match_phar_models(mols, num_clusters=6):
    # for multiple ref molecules, model need extract common pharmacophore model to screen the ligands
    # factory = Gobbi_Pharm2D.factory
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    all_features = np.array([list(fp) for fp in fps])
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(all_features)
    clustered_mols = [[] for _ in range(max(clusters) + 1)]
    for mol, label in zip(mols, clusters):
        clustered_mols[label].append(mol)
    selected_mols = []
    # 对每个类别中的图进行遍历
    for cluster_mol in clustered_mols:
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in cluster_mol]
        max_similarity = 0
        most_similar_index = 0

        for i in range(len(cluster_mol)):
            total_similarity = 0
            for j in range(len(cluster_mol)):
                if i != j:  # 忽略与自身比较
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    total_similarity += similarity
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                most_similar_index = i
        selected_mols.append(cluster_mol[most_similar_index])
    pp_graph_list= [mol2ppgraph(mol) for mol in selected_mols]  #get reference graph dataset
    save_graphs("/raid/yyw/PharmDiGress/data/LRRK2_pdb/lrrk2_phar_graphs.bin", pp_graph_list)
    # Calculate the 3D subgraph isomorphim matching score in every pair molecular graph
    return selected_mols, pp_graph_list

if __name__ == '__main__':
    smiles = 'CC1=C(C(C)=O)C(N(C(CC)CC)C2=NC(NC3=NC=C(N4CCNCC4)C=C3)=NC=C12)=O'
    mol = Chem.MolFromSmiles(smiles)
    g_, smiles_code_res = mol2ppgraph(mol)  ##g_药效团；smiles_code_res：smiles编码（矩阵格式）