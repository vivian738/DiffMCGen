import torch
from tqdm import tqdm
import pandas as pd

from src.datasets.pharmacophore_eval import match_score, get_best_match_phar_models
from dgl.data.utils import load_graphs
from rdkit import Chem
import csv
from src.eval import sascorer
from rdkit.Chem import QED, AllChem
from multiprocessing import Pool
import joblib
import os
def process_mol(args):
    mol, pp_graph_list, loaded_reg = args
    if mol is None:
        return (0, 0, 0, 0)  # 如果分子无效，返回默认值
    else:
        pharma_match_score_list = [match_score(mol, pp_graph) for pp_graph in pp_graph_list]
        return (
            max(pharma_match_score_list),
            sascorer.calculateScore(mol) * 0.1,
            QED.default(mol),
            loaded_reg.predict([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]).item()
        )
    
def cal_elpro(args):
    mol, pp_graph_list1, pp_graph_list2 = args
    data1 = max([match_score(mol, pp_graph1) for pp_graph1 in pp_graph_list1])
    data2 = max([match_score(mol, pp_graph2) for pp_graph2 in pp_graph_list2])
    return (data1, data2)

if __name__ == '__main__':
    # target_path = '/raid/yyw/PharmDiGress/data/LRRK2_pdb'
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
    # target = dataset_info.prop2idx[0]
    # pp_graph_list, _ = load_graphs(f"./data/{target}_pdb/{target}_phar_graphs.bin")
    # for pp_graph in pp_graph_list:
    #     pp_graph.ndata['h'] = \
    #         torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
    #     pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
    # loaded_reg = joblib.load('./data/stacking_regressor_model.pkl')
    suppl = Chem.SDMolSupplier('./data/csd/raw/CSD_process.sdf', removeHs=False, sanitize=True)
    # mols = [mol for mol in suppl][181336:]
    test_dataset = './data/csd/raw/csd_prop1.csv'
    # with open(test_dataset, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     for mol in tqdm(mols, total=len(mols)):
    #         arg = mol, pp_graph_list, loaded_reg
    #         result = process_mol(arg)
    #         writer.writerow(result)
        
    
    pp_graph_list1, _ = load_graphs("./data/HPK1_pdb/hpk1_phar_graphs.bin")
    for pp_graph1 in pp_graph_list1:
        pp_graph1.ndata['h'] = \
            torch.cat((pp_graph1.ndata['type'], pp_graph1.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph1.edata['h'] = pp_graph1.edata['dist'].reshape(-1, 1).float()
    pp_graph_list2, _ = load_graphs("./data/LRRK2_pdb/lrrk2_phar_graphs.bin")
    for pp_graph2 in pp_graph_list2:
        pp_graph2.ndata['h'] = \
            torch.cat((pp_graph2.ndata['type'], pp_graph2.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph2.edata['h'] = pp_graph2.edata['dist'].reshape(-1, 1).float()
    mols_csd = [mol for mol in suppl]
    with open(test_dataset, 'r', newline='') as csvfile1:
        reader = csv.reader(csvfile1)
        next(reader)
        with open('./data/csd/raw/csd_prop3.csv', 'a', newline='') as csvfile2:
            writer1 = csv.writer(csvfile2, delimiter=',')
            writer1.writerow(["pharma_score","SA","QED", "acute_tox", "glp1_score", 'cav32_score', 'hpk1_score', 'lrrk2_score'])
            # with Pool(8) as pool:
            #     for result in pool.imap(cal_elpro, tqdm([(mol, pp_graph_list1, pp_graph_list2) for mol in mols_csd], total=len(mols_csd))):
            #         writer1.writerow(result)
            for i, row in enumerate(tqdm(list(reader))):
                mol = mols_csd[i]
                d1 = max([match_score(mol, pp_graph1) for pp_graph1 in pp_graph_list1])
                row.append(d1)
                d2 = max([match_score(mol, pp_graph2) for pp_graph2 in pp_graph_list2])
                row.append(d2)
                writer1.writerow(row)
    # dataset = './data/moses/moses_pyg/raw/train_moses.csv'
    # with open(dataset, 'r', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     with open('./data/moses/moses_pyg/raw/moses_train1.csv', 'a', newline='') as csvfile1:
    #         writer_ = csv.writer(csvfile1, delimiter=',')
    #         # writer_.writerow(["smiles","pharma_score","SA","QED", "acute_tox", "glp1_score", 'cav32_score'])
    #         for rowr in tqdm(list(reader)[1216264:]):
    #             mol = Chem.MolFromSmiles(rowr[0])
    #             d1_ = max([match_score(mol, pp_graph1) for pp_graph1 in pp_graph_list1])
    #             rowr.append(d1_)
    #             d2_ = max([match_score(mol, pp_graph2) for pp_graph2 in pp_graph_list2])
    #             rowr.append(d2_)
    #             writer_.writerow(rowr)