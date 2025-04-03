# DiffMC-Gen: A Dual Denoising Diffusion Model for Multi-Conditional Molecular Generation

  - For the conditional generation experiments, check the `master` branch.

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n diffmc rdkit=2023.03.2 python=3.9```
  - `conda activate diffmc`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the dual model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=csd`. Look at `configs/dataset` for the list
of datasets that are currently available
    
## Checkpoints


  - the checkpoints of model train on MOSES, QM9, and CSD datasets were upload at checkpoints files.


## Generated samples

Set 'test_only' in configs.general as the path of trained checkpoints.


## Use DiffMC-Gen on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. You can base this file on `moses_dataset.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


## Citation
If you find this repo useful, please cite our related paper.

@article{https://doi.org/10.1002/advs.202417726,  
author = {Yang, Yuwei and Gu, Shukai and Liu, Bo and Gong, Xiaoqing and Lu, Ruiqiang and Qiu, Jiayue and Yao, Xiaojun and Liu, Huanxiang},  
title = {DiffMC-Gen: A Dual Denoising Diffusion Model for Multi-Conditional Molecular Generation},  
journal = {Advanced Science},  
volume = {n/a},  
number = {n/a},  
pages = {2417726},  
keywords = {deep learning, diffusion model, drug design, molecular generation, multi-objective optimization},  
doi = {https://doi.org/10.1002/advs.202417726}  
}
