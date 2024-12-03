from rdkit import Chem

import torch
from tqdm import tqdm
from torch_geometric.data import download_url
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import sascorer
from rdkit.Chem import QED, AllChem
from datasets.pharmacophore_eval import match_score
from dgl.data.utils import load_graphs
import multiprocessing

import joblib
import csv
def process_mol(args):
    smiles, pp_graph_list, loaded_reg = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (smiles, 0, 0, 0, 0)  # 如果分子无效，返回默认值
    else:
        pharma_match_score_list = [match_score(mol, pp_graph) for pp_graph in pp_graph_list]
        return (
            smiles,
            max(pharma_match_score_list),
            sascorer.calculateScore(mol) * 0.1,
            QED.default(mol),
            loaded_reg.predict([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]).item()
        )

if __name__ == '__main__':
    # train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    # train_path = download_url(train_url, '/raid/yyw/PharmDiGress/data/moses/moses_pyg/raw/')
    pp_graph_list, _ = load_graphs("/raid/yyw/PharmDiGress/data/PDK1_pdb/pdk1_phar_graphs.bin")
    for pp_graph in pp_graph_list:
        pp_graph.ndata['h'] = \
            torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
    loaded_reg = joblib.load('/raid/yyw/PharmDiGress/data/stacking_regressor_model_1.pkl')

    train_dataset = '/raid/yyw/PharmDiGress/data/moses/moses_pyg/raw/train.csv'
    train_data = pd.read_csv(train_dataset)
    
    with open('/raid/yyw/PharmDiGress/data/moses/moses_pyg/raw/moses_train.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["smiles","pharma_score","SA","QED", "acute_tox"])
        for smiles in tqdm(train_data['SMILES'].values, total=len(train_data['SMILES'].values)):
            result = process_mol((smiles, pp_graph_list, loaded_reg))
            writer.writerow(result) 
        writer.close()