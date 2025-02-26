# Rdkit import should be first, do not move it
from rdkit import Chem

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

import src.utils as utils
from src.metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.regressor import RegressorDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'graph_ddm_regressor', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(dataset_config)
    assert cfg.model.type == 'discrete'
    if dataset_config["name"] == 'qm9':
        from datasets import qm9_dataset
        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

    elif dataset_config.name == 'moses':
        from datasets import moses_dataset
        datamodule = moses_dataset.MosesDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
    elif dataset_config.name == 'csd':
        from datasets import csd_dataset
        datamodule = csd_dataset.CSDDataModule(cfg)
        dataset_infos = csd_dataset.CSDinfos(datamodule, cfg)
    else:
        raise ValueError("Dataset not implemented")
    datamodule.prepare_data()
    train_smiles = None

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 4}

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    if cfg.general.regressor:
        prop2idx_sub = {
            cfg.model.context[0]: dataset_infos.prop2idx[cfg.model.context[0]]
        }
        prop_norms = datamodule.train_dataset.compute_property_mean_mad(prop2idx_sub)
        prop_norms_val = datamodule.val_dataset.compute_property_mean_mad(prop2idx_sub)
    else:
        prop_norms, prop_norms_val = None, None

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features,
                    'property_norms': prop_norms, 'property_norms_val': prop_norms_val}

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    model = RegressorDiscrete(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_mae',
                                              save_last=True,
                                              save_top_k=-1,    # was 5
                                              mode='min',
                                              every_n_epochs=1)
        print("Checkpoints will be logged to", checkpoint_callback.dirpath)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if cfg.general.gpus > 0 and torch.cuda.is_available() else 'cpu',
                      devices=1 if cfg.general.gpus > 0 and torch.cuda.is_available() else None,
                      limit_train_batches=20,     # TODO: remove
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


if __name__ == '__main__':
    main()