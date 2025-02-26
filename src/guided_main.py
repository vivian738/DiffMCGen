import faulthandler
faulthandler.enable()
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
import os
# os.environ["NCCL_DEBUG"] = "INFO"   #debug use
os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['HYDRA_FULL_ERROR']='1'   #
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import random
# from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from pytorch_lightning.utilities.model_summary import ModelSummary

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.regressor import RegressorDiscrete

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'csd']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = moses_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config.name == 'csd':
            from datasets import csd_dataset
            datamodule = csd_dataset.CSDDataModule(cfg)
            dataset_infos = csd_dataset.CSDinfos(datamodule, cfg)
            train_smiles = csd_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)


        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
        if cfg.general.regressor:
            prop2idx_sub = {
                cfg.model.context[0]: dataset_infos.prop2idx[cfg.model.context[0]]
            }
            prop_norms = datamodule.train_dataset.compute_property_mean_mad(prop2idx_sub)
            prop_norms_val = datamodule.val_dataset.compute_property_mean_mad(prop2idx_sub)
        else:
            prop_norms, prop_norms_val =None,None

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features,
                        'property_norms': prop_norms, 'property_norms_val': prop_norms_val}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    cfg_pretrained, guidance_sampling_model = get_resume(cfg, model_kwargs)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        # cfg.model = cfg_pretrained.model
        cfg.guidance = {'use_guidance': True, 'lambda_guidance': 0.5}
    utils.create_folders(cfg)
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    guidance_model = RegressorDiscrete.load_from_checkpoint(os.path.join(root_dir, cfg.general.trained_regressor_path))
    model_kwargs['guidance_model'] = guidance_model

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                    #   profiler="simple",
                      strategy="ddp_find_unused_parameters_true", # Needed to load old checkpoints: ddp_find_unused_parameters_true
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      limit_test_batches=100,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                    #   fast_dev_run=5,  # debug: True-every process circulate 5 times, int-circulate {int} times
                      enable_progress_bar=cfg.train.progress_bar,
                    #   val_check_interval=cfg.general.val_check_interval,
                      logger=[])
                      #accumulate_grad_batches=4)

    model = guidance_sampling_model
    model.args = cfg
    model.guidance_model = guidance_model
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == '__main__':
    main()


