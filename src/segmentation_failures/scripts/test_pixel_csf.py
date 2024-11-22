"""
Similar to validate_pixel_csf.py, but execute the testing loop.
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
from segmentation_failures.scripts.validate_pixel_csf import setup_model

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


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
    # testing
    # ------------
    # Init callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.test.items():
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
    trainer.test(model, datamodule=data_module)

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
