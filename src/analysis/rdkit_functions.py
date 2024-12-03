import numpy as np
import torch
import re
import wandb
from datasets.pharmacophore_eval import mol2ppgraph, match_score
from eval import sascorer
from rdkit.Chem import QED, AllChem
import joblib
from dgl.data.utils import load_graphs
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"
from rdkit.Geometry import Point3D

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3, 'Ca': 2, 'Mg': 2,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': {0: [1, 2]}, 'Bi': {0: 3, 1: 5}, 'Se': {0: [2, 4], 1: 6}, 'Ba': 2}
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {0: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 12:2, 15: 3, 16: 2, 17: 1, 20:2, 35: 1, 53: 1, 56:2}
margin1, margin2, margin3 = 10, 5, 3


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, pp_graph_list, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles
        self.loaded_reg = joblib.load('/raid/yyw/PharmDiGress/data/stacking_regressor_model_1.pkl')
        self.pp_graph_list = pp_graph_list

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        mols = []
        for graph in generated:
            atom_types, edge_types, conformers = graph
            mol = build_molecule_with_partial_charges(atom_types, edge_types, conformers, self.dataset_info.atom_decoder)
            # smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None:
                    valid.append(smiles)
                    mols.append(largest_mol)
            except:
                pass
            

        return valid, len(valid) / len(generated), np.array(num_components), mols

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, num_components, valid_mols = self.compute_validity(generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")


        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        if len(valid) > 0:
            pharma_score = np.mean([max([match_score(mol, pp_graph) for pp_graph in self.pp_graph_list]) for mol in valid_mols])
            qed = np.mean([QED.default(mol) for mol in valid_mols])
            sa = np.mean([sascorer.calculateScore(mol) * 0.1 for mol in valid_mols])
            acute_tox = np.mean([self.loaded_reg.predict([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]).item() for mol in valid_mols])
        else:
            pharma_score = 0
            qed = 0
            sa = 0
            acute_tox = 0
        return ([validity, uniqueness, novelty], unique, [pharma_score, qed, sa, acute_tox],
                dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), valid)


def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, positions, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()] )
    
    try:
        mol = mol.GetMol()
        # Set coordinates
        positions = positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        if positions.size(0) == mol.GetNumAtoms():
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
                mol.AddConformer(conf)
    except ValueError as e:
        print("Can't kekulize molecule")
        mol = None
    return mol



def build_molecule_with_partial_charges(atom_types, edge_types, positions, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                if len(atomid_valence)==2:
                # assert len(atomid_valence) == 2
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if verbose:
                        print("atomic num of atom with a large valence", an)
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
                else:
                    continue
    try:
        mol = mol.GetMol()
        # Set coordinates
        positions = positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        if positions.size(0) == mol.GetNumAtoms():
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
                mol.AddConformer(conf)
    except ValueError as e:
        print("Can't kekulize molecule")
        mol = None
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol, connection=False):
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        if connection:
            mol_conn = connect_fragments(mol)
            # if mol_conn is not None:
            mol = mol_conn
            if mol is None:
                return None, no_correct
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            try:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                check_idx = 0
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    type = int(b.GetBondType())
                    queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                    if type == 12:
                        check_idx += 1
                queue.sort(key=lambda tup: tup[1], reverse=True)

                if queue[-1][1] == 12:
                    return None, no_correct
                elif len(queue) > 0:
                    start = queue[check_idx][2]
                    end = queue[check_idx][3]
                    t = queue[check_idx][1] - 1
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, bond_dict[t])
            except Exception as e:
                # print(f"An error occurred in correction: {e}")
                return None, no_correct
    return mol, no_correct

def select_atoms_with_available_valency(frag):
    return [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0]


def connect_fragments(mol):
    # Get the separate fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) < 2:
        return mol

    combined_mol = Chem.RWMol(frags[0])

    for frag in frags[1:]:
        # Select all atoms with available valency from both molecules
        atoms1 = select_atoms_with_available_valency(combined_mol)
        atoms2 = select_atoms_with_available_valency(frag)
        
        # Try to connect using all combinations of available valency atoms
        for atom1 in atoms1:
            for atom2 in atoms2:
                new_mol = try_to_connect_fragments(combined_mol, frag, atom1, atom2)
                if new_mol is not None:
                    # If a valid connection is made, update the combined molecule and break
                    combined_mol = new_mol
                    break
            else:
                # Continue if the inner loop didn't break (no valid connection found for atom1)
                continue
            # Break if the inner loop did break (valid connection found)
            break
        else:
            # If no valid connections could be made with any of the atoms, return None
            return None

    return combined_mol

def try_to_connect_fragments(combined_mol, frag, atom1, atom2):
    # Make copies of the molecules to try the connection
    trial_combined_mol = Chem.RWMol(combined_mol)
    trial_frag = Chem.RWMol(frag)
    
    # Add the new fragment to the combined molecule with new indices
    new_indices = {atom.GetIdx(): trial_combined_mol.AddAtom(atom) for atom in trial_frag.GetAtoms()}
    
    # Add the bond between the suitable atoms from each fragment
    trial_combined_mol.AddBond(atom1.GetIdx(), new_indices[atom2.GetIdx()], Chem.BondType.SINGLE)
    
    # Adjust the hydrogen count of the connected atoms
    for atom_idx in [atom1.GetIdx(), new_indices[atom2.GetIdx()]]:
        atom = trial_combined_mol.GetAtomWithIdx(atom_idx)
        num_h = atom.GetTotalNumHs()
        atom.SetNumExplicitHs(max(0, num_h - 1))
        
    # Add bonds for the new fragment
    for bond in trial_frag.GetBonds():
        trial_combined_mol.AddBond(new_indices[bond.GetBeginAtomIdx()], new_indices[bond.GetEndAtomIdx()], bond.GetBondType())
    
    # Convert to a Mol object and try to sanitize it
    new_mol = Chem.Mol(trial_combined_mol)
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol  # Return the new valid molecule
    except Chem.MolSanitizeException:
        return None  # If the molecule is not valid, return None

def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

use_rdkit = True


# def check_stability(atom_types, edge_types, dataset_info, debug=False,atom_decoder=None):
#     if atom_decoder is None:
#         atom_decoder = dataset_info.atom_decoder

#     n_bonds = np.zeros(len(atom_types), dtype='int')

#     for i in range(len(atom_types)):
#         for j in range(i + 1, len(atom_types)):
#             n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
#             n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
#     n_stable_bonds = 0
#     for atom_type, atom_n_bond in zip(atom_types, n_bonds):
#         possible_bonds = allowed_bonds[atom_decoder[atom_type]]
#         if type(possible_bonds) == int:
#             is_stable = possible_bonds == atom_n_bond
#         else:
#             is_stable = atom_n_bond in possible_bonds
#         if not is_stable and debug:
#             print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
#         n_stable_bonds += int(is_stable)

#     molecule_stable = n_stable_bonds == len(atom_types)
#     return molecule_stable, n_stable_bonds, len(atom_types)

def check_stability(positions, atom_type, edge_type, rdkit_mol, dataset_info):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info.atom_decoder
    edge_type[edge_type == 4] = 1.5
    edge_type[edge_type < 0] = 0
    valencies = torch.sum(edge_type, dim=-1).long()
    n_stable_bonds = 0
    mol_stable = True
    charges = [atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()]
    for i, (at, valency) in enumerate(zip(atom_type, valencies)):
        at = at.item()
        valency = valency.item()
        charge = charges[i]
        possible_bonds = allowed_bonds[atom_decoder[at]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        n_stable_bonds += int(is_stable)

    return mol_stable,\
           n_stable_bonds,\
           len(atom_type)

def get_bond_order(atom1, atom2, distance, check_exists=True):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond

def compute_molecular_metrics(molecule_list, train_smiles, dataset_info):
    """ molecule_list: (dict) """

    if not dataset_info.remove_h:
        print(f'Analyzing molecule stability...')

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            if len(mol) == 4:
                atom_types, edge_types, positions, rdmol = mol
                validity_results = check_stability(positions, atom_types, edge_types, rdmol, dataset_info)
                molecule_stable += int(validity_results[0])
                nr_stable_bonds += int(validity_results[1])
                n_atoms += int(validity_results[2])

            else:
                rdmol = build_molecule(atom_types, edge_types, positions, dataset_info.atom_decoder)
                validity_results = check_stability(positions, atom_types, edge_types, rdmol, dataset_info)
                molecule_stable += int(validity_results[0])
                nr_stable_bonds += int(validity_results[1])
                n_atoms += int(validity_results[2])
            
        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {'mol_stable': -1, 'atm_stable': -1}
    if dataset_info.name == 'qm9':
        pp_graph_list, _ = load_graphs(f"/raid/yyw/PharmDiGress/data/lrrk2_pdb/lrrk2_phar_graphs.bin")
    else:
        target = list(dataset_info.prop2idx.keys())[0].split('_')[0]
        pp_graph_list, _ = load_graphs(f"/raid/yyw/PharmDiGress/data/{target}_pdb/{target}_phar_graphs.bin")
    # Calculate the 3D subgraph isomorphim matching score in every pair molecular graph
    for pp_graph in pp_graph_list:
        pp_graph.ndata['h'] = \
            torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
    molecule_list_new=[]
    for moll in molecule_list:
        if len(moll) == 4:
            m = moll[0:-1]
        else:
            m = moll
        molecule_list_new.append(m)
    metrics = BasicMolecularMetrics(dataset_info, pp_graph_list, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list_new)
    all_smiles = rdkit_metrics[-1]
    if wandb.run:
        nc = rdkit_metrics[-2]
        dic = {'Validity': rdkit_metrics[0][0],
               'Uniqueness': rdkit_metrics[0][1], 'Novelty': rdkit_metrics[0][2],
               'Pharmacophore matching score': rdkit_metrics[2][0], 'QED': rdkit_metrics[2][1],
               'SA score': rdkit_metrics[2][2], 'acute toxicity score': rdkit_metrics[2][3],
               'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu']}
        wandb.log(dic)

    return validity_dict, rdkit_metrics, all_smiles, molecule_list_new
