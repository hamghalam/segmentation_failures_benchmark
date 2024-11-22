import os
import sys
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from segmentation_failures.data.datamodules.nnunet_module import NNunetDataModule
from segmentation_failures.evaluation.segmentation.compute_seg_metrics import (
    compute_metrics_for_prediction_dir,
)
from segmentation_failures.utils.io import save_json

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


def setup_model(cfg: DictConfig) -> pl.LightningModule:
    # initialize segmentation backbone
    seg_backbone = hydra.utils.instantiate(cfg.backbone.hparams)
    if cfg.backbone.checkpoint is not None:
        # Not sure if this is a case I want to allow
        raise NotImplementedError
    model = hydra.utils.instantiate(cfg.segmentation.hparams, backbone=seg_backbone)
    return model


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
    logger.remove()  # Remove default 'stderr' handler
    logger.add(sys.stderr, level=config.loguru.level)
    logger.add(Path(config.paths.output_dir) / config.loguru.file, level=config.loguru.level)
    logger.info(os.getcwd())
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

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
    logger.info("Instantiating segmentation model")
    model = setup_model(config)

    # ------------
    # training
    # ------------
    # Init lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.train_seg.items():
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
        callbacks=callbacks,
        logger=expt_logger,
        _convert_="partial",
    )

    if config.trainer.get("auto_lr_find", False):
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
        print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lrfind.png")
    else:
        logger.info("Starting training...")
        trainer.fit(model, datamodule=data_module, ckpt_path=config.resume_from_checkpoint.path)
        logger.info("Training finished!")

    # Save configuration diff at the end of training to capture any runtime changes
    final_config_yaml = yaml.dump(OmegaConf.to_container(config), sort_keys=False)
    hydra_config_path = Path(config.paths.output_dir) / ".hydra/config.yaml"
    hydra_config_path.rename(hydra_config_path.parent / "initial_config.yaml")
    with open(hydra_config_path, "w") as file:
        file.write(final_config_yaml)

    if config.trainer.accelerator == "gpu":
        torch.cuda.empty_cache()
    # special case for nnunet datamodule: run a proper validation at the end
    if isinstance(data_module, NNunetDataModule):
        logger.info("Running validation on full images")
        data_module: NNunetDataModule = hydra.utils.instantiate(
            config.datamodule,
            preproc_only=True,
            batch_size=1,
        )
        trainer.validate(datamodule=data_module, ckpt_path="last")
        logger.info("Computing segmentation metrics")
        label_file_list = data_module.get_val_data_label_paths()
        metrics_dict, multimetrics_dict = compute_metrics_for_prediction_dir(
            metric_list=["dice"],
            prediction_dir=config.paths.predictions_dir,
            label_file_list=label_file_list,
            dataset_id=int(config.dataset.dataset_id),
            num_processes=3,
        )
        # save metrics
        case_ids = [p.name.split(".")[0] for p in label_file_list]
        save_dict = {}
        for m, metric_arr in metrics_dict.items():
            curr_dict = {case_id: metric_arr[i].tolist() for i, case_id in enumerate(case_ids)}
            save_dict[m] = {k: curr_dict[k] for k in sorted(curr_dict)}
        for m, metric_arr in multimetrics_dict.items():
            if m in save_dict:
                m = f"multiclass_{m}"
            curr_dict = {case_id: metric_arr[i].tolist() for i, case_id in enumerate(case_ids)}
            save_dict[m] = {k: curr_dict[k] for k in sorted(curr_dict)}
        save_json(save_dict, Path(config.paths.output_dir) / "validation_results.json")
        logger.info("Validation finished!")
    with open(Path(config.paths.output_dir) / "COMPLETED", "w") as file:
        file.write("")


if __name__ == "__main__":
    main()
