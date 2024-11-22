import sys
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from segmentation_failures.data.datamodules.quality_regression import (
    QualityRegressionDataModule,
)
from segmentation_failures.data.datamodules.vae import VAEdataModule
from segmentation_failures.utils import GLOBAL_SEEDS
from segmentation_failures.utils.checkpointing import (
    get_checkpoint_from_experiment,
    get_experiments_for_seed_fold,
)

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


class NotTrainableException(Exception):
    """Communicates that a module is not trainable."""

    def __init__(self, message="This module is not trainable. Configuration error?"):
        super().__init__(message)


def setup_segmentation_model(cfg: DictConfig, load_best_ckpt=False):
    if cfg.backbone.checkpoint is not None:
        # load ckpt. Not sure how to do that yet. This is only needed when loading external
        # models. In that case, I need to write a wrapper and configure it in `cfg.segmentation`
        raise NotImplementedError
    seg_backbone = hydra.utils.instantiate(cfg.backbone.hparams)
    seg_model = hydra.utils.instantiate(cfg.segmentation.hparams, backbone=seg_backbone)
    ensemble = False
    if "csf_pixel" in cfg and "DeepEnsemble" in cfg.csf_pixel.hparams["_target_"]:
        ensemble = True
    if cfg.segmentation.checkpoint is None:
        seg_expt_name = "-".join(cfg.expt_name.split("-")[:2])
        seg_expt_root = Path(cfg.paths.output_dir).parents[2] / seg_expt_name / "train_seg"
        # structure is expt_root/expt_name/{train_image_csf,test_fd}/output_dir
        found_expts = get_experiments_for_seed_fold(
            seg_expt_root,
            list(GLOBAL_SEEDS.values()) if ensemble else cfg.seed,  # TODO hard-coded -> not good
            cfg.datamodule.fold,
        )
        assert (
            len(found_expts) < 6
        ), f"I use max 5 models for ensembling atm. Found expts: {found_expts}"
        if len(found_expts) == 0:
            raise ValueError(
                f"Found no segmentation experiments that match seed and fold in {seg_expt_root}."
            )
        if len(found_expts) > 1 and not ensemble:
            raise ValueError(
                f"Found {len(found_expts)} segmentation experiments in {seg_expt_root} but ensemble was not configured."
            )
        checkpoints = []
        for expt_dir in found_expts:
            checkpoints.append(
                str(get_checkpoint_from_experiment(expt_dir, last_ckpt=not load_best_ckpt))
            )
        if len(checkpoints) == 1:
            cfg.segmentation.checkpoint = checkpoints[0]
        else:
            cfg.segmentation.checkpoint = checkpoints
        logger.info(f"Automatically determined {len(checkpoints)} checkpoint path(s)")
    elif cfg.segmentation.checkpoint == "DEBUG_NONE":
        logger.warning(
            "Configured no checkpoint for debugging purposes. Using randomly initialized network."
        )
        return seg_model
    # load the weights
    checkpoints = cfg.segmentation.checkpoint
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]
    logger.info("Loading checkpoint(s):\n" + "\n".join(checkpoints))
    # a bit ugly, but for now it works
    model_list = []
    for ckpt_path in checkpoints:
        # Need to re-initialize/deepcopy the backbone object, otherwise it will be
        # shared between segmentation models.
        seg_backbone = hydra.utils.instantiate(cfg.backbone.hparams)
        model_list.append(
            seg_model.__class__.load_from_checkpoint(
                ckpt_path, backbone=seg_backbone, **cfg.segmentation.hparams
            )
        )
    if len(model_list) == 1:
        return model_list[0]
    return model_list


def setup_model(cfg: DictConfig) -> pl.LightningModule:
    if cfg.get("csf_aggregation"):
        assert (
            cfg.get("csf_image") is None
        ), "Can either have image-level CSF or pixel-level + aggregation!"
        assert "csf_pixel" in cfg, "Pixel-CSF is needed for aggregation"
        if not cfg.csf_aggregation.trainable:
            raise NotTrainableException
        if not cfg.csf_aggregation.twostage:
            # the method needs the pixel csf instance
            seg_model = setup_segmentation_model(cfg, load_best_ckpt=False)
            pixel_csf = hydra.utils.instantiate(cfg.csf_pixel.hparams, segmentation_net=seg_model)
            if cfg.csf_pixel.checkpoint is not None:
                # here I need to extract the network from the lightning checkpoint.
                raise NotImplementedError(
                    "So far I don't have any methods with trained pixel csf."
                )
            # initialize confidence_aggr using the pixel_csf
            image_csf = hydra.utils.instantiate(cfg.csf_aggregation.hparams, pixel_csf=pixel_csf)
        else:
            image_csf = hydra.utils.instantiate(cfg.csf_aggregation.hparams)
    else:
        assert cfg.get("csf_image") is not None
        if not cfg.csf_image.trainable:
            raise NotTrainableException
        if cfg.csf_image.needs_pretrained_segmentation:
            seg_model = setup_segmentation_model(cfg, load_best_ckpt=False)
            image_csf = hydra.utils.instantiate(cfg.csf_image.hparams, segmentation_net=seg_model)
        else:
            image_csf = hydra.utils.instantiate(cfg.csf_image.hparams)
    return image_csf


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
    logger.remove()  # Remove default 'stderr' handler
    logger.add(sys.stderr, level=config.loguru.level)
    logger.add(Path(config.paths.output_dir) / config.loguru.file, level=config.loguru.level)

    logger.info(f"Experiment directory: {config.paths.output_dir}")

    resume_from_ckpt = config.resume_from_checkpoint.path
    if resume_from_ckpt is not None and config.resume_from_checkpoint.load_expt_config:
        # load the configuration from this experiment if possible
        ckpt_path = Path(config.resume_from_checkpoint.path)
        hydra_config_path = ckpt_path.parents[1] / ".hydra/config.yaml"
        if not hydra_config_path.exists():
            raise FileNotFoundError(
                f"Could not find configuration file {hydra_config_path} for checkpoint {ckpt_path}."
            )
        logger.info(f"Loading configuration from {hydra_config_path}")
        old_config = config
        with open(hydra_config_path, "r") as file:
            config = OmegaConf.load(file)
        # merge the configurations: the paths and expt_* should not be changed
        for k in ["expt_name", "expt_group", "resume_from_checkpoint", "paths"]:
            config[k] = old_config[k]

    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    # ------------
    # data
    # ------------
    logger.info(f"Instantiating datamodule <{config.datamodule['_target_']}>")
    # Special for quality regression: need to overwrite the hparams
    data_module: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    data_module.prepare_data()
    # If another split should be used for csf training than for segmentation model training,
    # this can be done through the configuration.
    if hasattr(data_module, "preprocess_info") and config.datamodule.get("spacing", []) is None:
        # workaround. I dislike this solution
        config.datamodule.spacing = data_module.preprocess_info["spacing"]
    # ------------
    # model
    # ------------
    if isinstance(data_module, QualityRegressionDataModule):
        config.csf_image.hparams.output_names = data_module.metric_target_names
        config.csf_image.hparams.img_size = data_module.dataset_fingerprint["img_size"]
        config.csf_image.hparams.voxel_spacing = data_module.dataset_fingerprint["spacing"]
        config.csf_image.hparams.blocks_per_stage = config.backbone.hparams.get(
            "blocks_per_stage", 1
        )
    elif isinstance(data_module, VAEdataModule):
        config.csf_image.hparams.img_size = data_module.img_size
    elif "csf_aggregation" in config and (
        "Heuristic" in config.csf_aggregation.hparams["_target_"]
        or "Radiomics" in config.csf_aggregation.hparams["_target_"]
    ):
        config.csf_aggregation.hparams.target_metrics = data_module.metric_target_names
    model = setup_model(config)

    # ------------
    # training
    # ------------
    # Init lightning callbacks
    callbacks = []
    if config.callbacks.train is not None:
        for _, cb_conf in config.callbacks.train.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf['_target_']}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    expt_logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_conf['_target_']}>")
                expt_logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer['_target_']}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        _convert_="partial",
        callbacks=callbacks,
        logger=expt_logger,
    )

    logger.info("Starting training...")
    if config.get("auto_lr_find", False):
        tuner = pl.tuner.tuning.Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=data_module)
        print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lrfind.png")
    else:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_ckpt)
        logger.info("Training finished!")

        # validation
        ckpt_path = resume_from_ckpt
        if ckpt_path is None:
            ckpt_callback = [c for c in callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)]
            ckpt_path = "last"
            if len(ckpt_callback) > 0 and ckpt_callback[0].save_top_k > 0:
                ckpt_path = "best"
        trainer.validate(
            model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
        logger.info("Validation finished!")

    # Save configuration diff at the end of training to capture any runtime changes
    final_config_yaml = yaml.dump(OmegaConf.to_container(config), sort_keys=False)
    hydra_config_path = Path(config.paths.output_dir) / ".hydra/config.yaml"
    hydra_config_path.rename(hydra_config_path.parent / "initial_config.yaml")
    with open(hydra_config_path, "w") as file:
        file.write(final_config_yaml)
    with open(Path(config.paths.output_dir) / "COMPLETED", "w") as file:
        file.write("")


if __name__ == "__main__":
    main()
