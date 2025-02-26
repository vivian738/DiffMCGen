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

import io
import PIL
from PIL import ImageFont
from PIL import ImageDraw, Image
from rdkit.Chem import PyMol
from PIL import ImageFile

from src.analysis.rdkit_functions import build_xae_molecule

ImageFile.LOAD_TRUNCATED_IMAGES = True
allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6], 'Ba':[2]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

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

        # add atoms to mol and keep track of index
        node_to_idx = {}
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
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"failed: {e}")
            mol = None
        
        if positions is not None and np.isnan(positions).any()==False:
            X, A, E = build_xae_molecule(torch.tensor(positions), torch.tensor(node_list), self.dataset_infos)
            mol_3 = Chem.RWMol()
            for atom in X:
                a = Chem.Atom(atom_decoder[atom.item()])
                mol_3.AddAtom(a)

            all_bonds = torch.nonzero(A)
            for bond in all_bonds:
                if bond[0].item() != bond[1].item():
                    mol_3.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
            conf = Chem.Conformer(mol_3.GetNumAtoms())   #可能顺序不一样
            for i in range(mol_3.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
            mol_3.AddConformer(conf)
        else:
            mol_3 = None

        return mol, mol_3

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph', conformer2d=None,
                  file_prefix='molecule'): #conformer_list: list, 
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol_2, mol_3 = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy(), molecules[i][2].numpy())
            if mol_2 is not None:
                self.plot_save_molecule(molecules[i], mol_2, save_path=file_path, conformer2d=conformer2d)
                if log is not None and wandb.run:
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            elif mol_3 is not None:
                self.plot_save_molecule(molecules[i], mol_3, save_path=file_path, conformer2d=conformer2d)

                if log is not None and wandb.run:
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
        
    
    def plot_save_molecule(self, graph_list, mol, save_path, conformer2d=None):
        buffer = io.BytesIO()
        new_im = PIL.Image.new('RGB', (600, 300), color='white')
        pil3d, max_dist = self.generatePIL3d(graph_list, buffer)
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
        draw.text((420, 15), f"3D view. Diam={max_dist:.1f}", font=font, fill='black')
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


    def generatePIL3d(self, mol, buffer, alpha=1.):
        # pymol = PyMol.MolViewer()
        # pymol.ShowMol(mol)
        # tmp_png = 'tmp.png'
        # png_data = pymol.GetPNG()
        # png_data.save(tmp_png)
        # try:
        #     pil_image = PIL.Image.open(tmp_png)
        # except PIL.UnidentifiedImageError:
        #     pil_image=PIL.Image.new('RGB', (300, 300), (255,255,255))
        atom_types = mol[0]
        edge_types = mol[1]
        positions = mol[2]
        num_atom_types = len(self.dataset_infos.atom_decoder)
        white = (1, 1, 1)
        hex_bg_color = '#000000' #'#666666'

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal', adjustable='datalim')
        ax.view_init(elev=90, azim=-90)
        ax.set_facecolor(white)
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax._axis3don = False

        ax.w_xaxis.line.set_color("white")

        # max_value = positions.abs().max().item()
        axis_lim = 0.7
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)

        max_dist = self.plot_molecule3d(ax, positions, atom_types, edge_types, alpha, hex_bg_color, num_atom_types)

        plt.tight_layout()
        plt.savefig(buffer, format='png', pad_inches=0.0)
        pil_image = PIL.Image.open(buffer)
        plt.close()
        return pil_image, max_dist
    
    def plot_molecule3d(self, ax, positions, atom_types, edge_types, alpha, hex_bg_color, num_atom_types):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Normalize the positions for plotting
        max_x_dist = x.max() - x.min()
        max_y_dist = y.max() - y.min()
        max_z_dist = z.max() - z.min()
        max_dist = max(max_x_dist, max_y_dist, max_z_dist) / 1.8
        x_center = (x.min() + x.max()) / 2
        y_center = (y.min() + y.max()) / 2
        z_center = (z.min() + z.max()) / 2
        x = (x - x_center) / max_dist
        y = (y - y_center) / max_dist
        z = (z - z_center) / max_dist

        radii = 0.4
        areas = 300 * (radii ** 2)
        if num_atom_types == 4:
            colormap = ['k', 'b', 'r', 'c']             # QM9 no H
        elif num_atom_types == 5:
            colormap = ['C7', 'k', 'b', 'r', 'c']
        elif num_atom_types == 14:
            colormap = ['C7', 'C0', 'k', 'b', 'r', 'c', 'C1', 'C2', 'C3', 'y', 'C5', 'C6', 'C8', 'C9']
        elif num_atom_types == 13:
            colormap = ['C0', 'k', 'b', 'r', 'c', 'C1', 'C2', 'C3', 'y', 'C5', 'C6', 'C8', 'C9']
        else:
            colormap = [f'C{a}' for a in range(num_atom_types)]

        colors = [colormap[a] for a in atom_types]
        for i in range(edge_types.shape[0]):
            for j in range(i + 1, edge_types.shape[1]):
                draw_edge = edge_types[i, j]
                if draw_edge > 0:
                    ax.plot([x[i].cpu().numpy(), x[j].cpu().numpy()],
                            [y[i].cpu().numpy(), y[j].cpu().numpy()],
                            [z[i].cpu().numpy(), z[j].cpu().numpy()],
                            linewidth=1, c=hex_bg_color, alpha=alpha)

        ax.scatter(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), s=areas, alpha=0.9 * alpha, c=colors)
        return max_dist
