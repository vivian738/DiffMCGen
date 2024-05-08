import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from analysis.rdkit_functions import correct_mol, build_molecule, mol2smiles

import io
import PIL
from PIL import ImageFont
from PIL import ImageDraw, Image
from rdkit.Chem import PyMol
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix, positions):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()
        node_to_idx = {}
        # add atoms to mol and keep track of index
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None
        
        # Set coordinates
        # positions = positions.double()
        if positions is not None:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
            mol.AddConformer(conf)

        # mol, no_correct = correct_mol(mol)
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph', conformer2d=None,
                  file_prefix='molecule'): #conformer_list: list, 
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        valid_molecules = []
        for i, graph in enumerate(molecules):
            atom_types, edge_types, conformers, _ = graph
            mol = build_molecule(atom_types, edge_types, conformers, self.dataset_infos.atom_decoder)
            if mol2smiles(mol) is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    valid_molecules.append(largest_mol)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    valid_molecules.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    valid_molecules.append(None)
                except Chem.rdchem.AtomKekulizeException:
                    valid_molecules.append(None)
            else:
                continue
        
                # visualize the final molecules
        if num_molecules_to_visualize == -1:
            num_molecules_to_visualize = len(valid_molecules)
            print(f"Visualizing {num_molecules_to_visualize} of {len(valid_molecules)}")
        if num_molecules_to_visualize > len(valid_molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(valid_molecules)

        all_file_paths = []
        for i in range(num_molecules_to_visualize):
            if valid_molecules[i] is None:
                continue
            else:
                file_path = os.path.join(path, f'{file_prefix}_{i}.png')
                self.plot_save_molecule(valid_molecules[i], save_path=file_path, conformer2d=conformer2d) #conformers=conformer_list[i]
                all_file_paths.append(file_path)

                if log is not None and wandb.run:
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
        
        return all_file_paths


    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = []
        for i in range(nodes_list.shape[0]):
            mols.append(self.mol_from_graphs(nodes_list[i], adjacency_matrix[i], positions=None))
        
        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append([positions.x, positions.y, positions.z])
        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}_grid_image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols
    
    def plot_save_molecule(self, mol, save_path, conformer2d=None):
        buffer = io.BytesIO()
        new_im = PIL.Image.new('RGB', (600, 300), color='white')
        pil3d = self.generatePIL3d(mol, buffer)
        pil3d = pil3d.resize((300, 300))
        new_im.paste(pil3d, (300, 0, 600, 300))
        # f_3d = []
        # for conformer in conformers:
        #     pil3d = self.generatePIL3d(mol, buffer, conformer)
        #     pil3d = pil3d.resize((300, 300))
        #     f_3d.append(pil3d)
        # new_im.paste(f_3d[0], (300, 0, 600, 300))
        # new_im.paste(f_3d[1], (600, 0, 900, 300))
        # new_im.paste(f_3d[2], (0, 300, 300, 600))
        # new_im.paste(f_3d[3], (300, 300, 600, 600))
        # new_im.paste(f_3d[4], (600, 300, 900, 600))
        try:
            pil2d = self.generatePIL2d(mol, conformer2d)
            new_im.paste(pil2d, (0, 0, 300, 300))
        except ValueError:
            print("Value error in generate PIL2D. The ")
            return

        draw = ImageDraw.Draw(new_im)
        real_path = os.path.realpath(__file__)
        dir_path = os.path.dirname(real_path)
        try:        # This normally works but sometimes randomly crashes
            font = ImageFont.truetype(os.path.join(dir_path, "Arial.ttf"), 15)
        except OSError:
            font = ImageFont.load_default()
        draw.text((420, 15), "3D view", font=font, fill='black')
        draw.text((100, 15), "2D view", font=font, fill='black')
        new_im.save(save_path, "PNG")
        buffer.close()

    def generatePIL2d(self, mol, conformer2d=None):
        """ mol: RdKit molecule object
            conformer2d: n x 3 tensor defining the coordinates which should be used to plot (used for chains vis). """
        # if mol is not None:
        if conformer2d is None:
            AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        #  Aligh all the molecules
        if conformer2d is not None:
            conformer2d = conformer2d.double()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = conformer2d[j, 0].item(), conformer2d[j, 1].item(), conformer2d[j, 2].item()

                conf.SetAtomPosition(j, Point3D(x, y, z))
        return Draw.MolToImage(mol)
        # else:
        #     return PIL.Image.new('RGB', (300, 300), (255,255,255))


    def generatePIL3d(self, mol, buffer):
        # try:
        #     Chem.SanitizeMol(mol[-1])
        # positions = positions.double()
        # conf = Chem.Conformer(mol.GetNumAtoms())
        # if positions.size(0) == mol.GetNumAtoms():
        #     for i in range(mol.GetNumAtoms()):
        #         conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        #         mol.AddConformer(conf)
        pymol = PyMol.MolViewer()
        pymol.ShowMol(mol)
        tmp_png = 'tmp.png'
        png_data = pymol.GetPNG()
        png_data.save(tmp_png)
        try:
            pil_image = PIL.Image.open(tmp_png)
        except PIL.UnidentifiedImageError:
            pil_image=PIL.Image.new('RGB', (300, 300), (255,255,255))

        return pil_image
