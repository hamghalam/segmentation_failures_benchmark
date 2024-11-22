"""
Similar to evaluate.py, but I don't use pytorch-lightning here. This script is the one I currently use most.

Inputs: test data, model checkpoint, confidence scoring function, flag which of prediction/evaluation to do
Outputs: Saves segmentation mask + per-sample confidences

- Loop over test set and predict each sample
- For methods that compute confidences from softmax output: just implement it as a simple function/class?
- For methods that compute confidences based on intermediate network activations:
not sure what's the best way to extract feature maps
- For methods that include confidence scoring components in training: need to modify lightning module.
Maybe the predict_step method can be used to separate the inference behavior?

"""

import sys
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from segmentation_failures.scripts.train_image_csf import setup_segmentation_model

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


def setup_model(cfg: DictConfig, seg_model) -> pl.LightningModule:
    if cfg.get("csf_pixel") is None:
        logger.info("No pixel confidence configured. Continuing with segmentation model")
        return seg_model
    # initialize pixel csf using the segmentation network
    pixel_csf = hydra.utils.instantiate(cfg.csf_pixel.hparams, segmentation_net=seg_model)
    if cfg.csf_pixel.checkpoint is not None:
        # here I need to extract the network from the lightning checkpoint.
        raise NotImplementedError("So far I don't have any methods with trained pixel csf.")
    return pixel_csf


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
    logger.remove()  # Remove default 'stderr' handler
    logger.add(sys.stderr, level=config.loguru.level)
    logger.add(Path(config.paths.output_dir) / config.loguru.file, level=config.loguru.level)

    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    if config.get("image_csf") is not None or config.get("csf_aggregation") is not None:
        raise ValueError("This script is only for methods without image-csf or csf-aggregation")

    logger.info(f"Experiment directory: {config.paths.output_dir}")
    # ------------
    # data
    # ------------
    logger.info(f"Instantiating datamodule <{config.datamodule['_target_']}>")
    if (
        config.datamodule["_target_"]
        == "segmentation_failures.data.datamodules.nnunet_module.NNunetDataModule"
    ):
        # inference-style validation
        config.datamodule.preproc_only = True
        config.datamodule.batch_size = 1
    data_module: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    data_module.prepare_data()
    if hasattr(data_module, "preprocess_info"):
        # workaround. I dislike this solution
        config.datamodule.spacing = data_module.preprocess_info["spacing"]
    # ------------
    # model
    # ------------
    logger.info("Instantiating model")
    seg_model = setup_segmentation_model(config, load_best_ckpt=False)
    model = setup_model(config, seg_model)

    # ------------
    # validation
    # ------------
    # Init callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.validate.items():
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

    logger.info(f"Instantiating trainer <{config.trainer['_target_']}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        _convert_="partial",
        callbacks=callbacks,
        logger=expt_logger,
    )

    logger.info("Starting validation...")
    trainer.validate(model, datamodule=data_module)

    # Save configuration diff at the end to capture any runtime changes
    final_config_yaml = yaml.dump(OmegaConf.to_container(config), sort_keys=False)
    hydra_config_path = Path(config.paths.output_dir) / ".hydra/config.yaml"
    hydra_config_path.rename(hydra_config_path.parent / "initial_config.yaml")
    with open(hydra_config_path, "w") as file:
        file.write(final_config_yaml)
    with open(Path(config.paths.output_dir) / "COMPLETED", "w") as file:
        file.write("")
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()
