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

from segmentation_failures.data.datamodules.quality_regression import (
    QualityRegressionDataModule,
)
from segmentation_failures.data.datamodules.vae import VAEdataModule
from segmentation_failures.evaluation import (
    ExperimentData,
    evaluate_failures,
    evaluate_ood,
)
from segmentation_failures.scripts.train_image_csf import setup_segmentation_model
from segmentation_failures.utils.checkpointing import (
    get_checkpoint_from_experiment,
    get_experiments_for_seed_fold,
)

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


# ugly af, but fastest solution for now
# this allows for example quality regression without confidence mask on ensemble predictions
def check_alternative_pixel_csf(cfg):
    assert cfg.expt_name.count("-") == 3
    expt_name_parts = cfg.expt_name.split("-")
    pixel_csf = expt_name_parts[2]
    img_csf = expt_name_parts[3]
    if pixel_csf in ["mcdropout", "deep_ensemble"] and img_csf in [
        "quality_regression",
        "vae_image_and_mask",
        "vae_mask_only",
    ]:
        # the image csf check is a safety measure
        expt_name_parts[2] = "baseline"
    else:
        return None
    expt_name = "-".join(expt_name_parts)
    return Path(cfg.paths.output_dir).parents[2] / expt_name / "train_image_csf"


def determine_training_ckpt(cfg):
    # Determine which image CSF checkpoint to use
    train_expt_path = Path(cfg.paths.output_dir).parents[1] / "train_image_csf"
    if not train_expt_path.exists():
        # try to look for a training run without pixel_csf
        train_expt_path = check_alternative_pixel_csf(cfg)
    found_expts = get_experiments_for_seed_fold(
        train_expt_path,
        seed=cfg.seed,
        fold=cfg.datamodule.fold,
    )
    if len(found_expts) > 1:
        logger.warning(
            f"Found {len(found_expts)} segmentation experiments that match seed and fold. Selecting the latest version."
        )
    ckpt = get_checkpoint_from_experiment(sorted(found_expts)[-1], last_ckpt=cfg.test.last_ckpt)
    logger.info(f"Auto-determined the CSF checkpoint to use: {ckpt}")
    return ckpt


def setup_image_csf_model(cfg: DictConfig, seg_model) -> pl.LightningModule:
    if cfg.get("csf_aggregation"):
        # methods with pixel-csf & aggregation
        assert (
            cfg.get("csf_image") is None
        ), "Can either have image-level CSF or pixel-level + aggregation!"
        assert "csf_pixel" in cfg, "Pixel-CSF is needed for aggregation"
        if cfg.csf_aggregation.twostage:
            image_csf = hydra.utils.instantiate(cfg.csf_aggregation.hparams)
        else:
            pixel_csf = hydra.utils.instantiate(cfg.csf_pixel.hparams, segmentation_net=seg_model)
            if cfg.csf_pixel.checkpoint is not None:
                # here I need to extract the network from the lightning checkpoint.
                raise NotImplementedError(
                    "So far I don't have any methods with trained pixel csf."
                )
            image_csf = hydra.utils.instantiate(cfg.csf_aggregation.hparams, pixel_csf=pixel_csf)
        # finally, load the actual checkpoint
        if cfg.csf_aggregation.trainable:
            ckpt_path = cfg.csf_aggregation.checkpoint
            if ckpt_path is None:
                ckpt_path = determine_training_ckpt(cfg)
                cfg.csf_aggregation.checkpoint = str(ckpt_path)
            if cfg.csf_aggregation.twostage:
                image_csf = image_csf.__class__.load_from_checkpoint(ckpt_path)
            else:
                image_csf = image_csf.__class__.load_from_checkpoint(
                    ckpt_path, pixel_csf=pixel_csf
                )
    else:
        # methods with image-csf
        assert cfg.get("csf_image") is not None
        kwargs = {}
        if cfg.csf_image.needs_pretrained_segmentation:
            kwargs = {"segmentation_net": seg_model}
        image_csf = hydra.utils.instantiate(cfg.csf_image.hparams, **kwargs)
        # finally, load the actual checkpoint
        if cfg.csf_image.trainable:
            ckpt_path = cfg.csf_image.checkpoint
            if ckpt_path is None:
                ckpt_path = determine_training_ckpt(cfg)
                cfg.csf_image.checkpoint = str(ckpt_path)
            image_csf = image_csf.__class__.load_from_checkpoint(ckpt_path, **kwargs)
    return image_csf


def get_previous_stage_expt(config):
    # assume output directory structure like this:
    # DatasetXXX/runs/
    # |- previous_stage_expt/
    #    |- test_fd/
    #      |- version1/
    #        |- predictions <- I need this
    #        |- confidence_maps <- I need this
    # |- this_expt_name/
    #    |- train_image_csf/
    #    |- test_fd/
    #       |- version1/  <- this is the output_dir
    runs_root_dir = Path(config.paths.output_dir).parents[2]
    # assume the expt name is backbone-segmentation-pixelcsf-imagecsf
    previous_stage_expt_name = "-".join(config.expt_name.split("-")[:-1])
    previous_stage_expt_root = runs_root_dir / previous_stage_expt_name / "test_pixel_csf"
    # get the prediction dir and confidence map dir and return them
    candidates = get_experiments_for_seed_fold(
        previous_stage_expt_root, config.seed, config.datamodule.fold
    )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No experiments found for seed {config.seed} and fold {config.datamodule.fold}."
        )
    else:
        # Use the latest experiment
        previous_stage_expt_dir = sorted(candidates)[-1]
        logger.info(
            f"Multiple experiments found for seed {config.seed} and fold {config.datamodule.fold}."
        )
    logger.info(f"Using the latest experiment: {previous_stage_expt_dir}.")
    return previous_stage_expt_dir


def get_previous_stage_paths(config):
    previous_stage_expt_dir = get_previous_stage_expt(config)
    # The more elegant way would be to use the config from the previous stage,
    # but I have issues with the hydra resolver, which I don't want to overwrite.
    path_dict = {
        "predictions_dir": str(previous_stage_expt_dir / Path(config.paths.predictions_dir).name),
        "pixel_confid_dir": str(
            previous_stage_expt_dir / Path(config.paths.pixel_confid_dir).name
        ),
        "prediction_samples_dir": str(
            previous_stage_expt_dir / Path(config.paths.prediction_samples_dir).name
        ),
        "results_dir": str(previous_stage_expt_dir / Path(config.paths.results_dir).name),
    }
    return path_dict


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
    logger.remove()  # Remove default 'stderr' handler
    logger.add(sys.stderr, level=config.loguru.level)
    logger.add(Path(config.paths.output_dir) / config.loguru.file, level=config.loguru.level)

    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)
    logger.info(f"Experiment directory: {config.paths.output_dir}")
    # ------------
    # data
    # ------------
    twostage = False
    if ("csf_image" in config and config.csf_image.get("twostage", False)) or (
        "csf_aggregation" in config and config.csf_aggregation.get("twostage", False)
    ):
        twostage = True
        prev_stage_paths = get_previous_stage_paths(config)
        logger.info(f"Using results from previous stage:\n{prev_stage_paths}")
        config.paths.predictions_dir = prev_stage_paths["predictions_dir"]
        config.paths.pixel_confid_dir = prev_stage_paths["pixel_confid_dir"]
        config.paths.prediction_samples_dir = prev_stage_paths["prediction_samples_dir"]
        if "results_saver" in config.callbacks.test:
            # I need to save the predictions of the first stage
            config.callbacks.test.results_saver.previous_stage_results_path = prev_stage_paths[
                "results_dir"
            ]
    logger.info(f"Instantiating datamodule <{config.datamodule['_target_']}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    data_module.prepare_data()
    if hasattr(data_module, "preprocess_info") and config.datamodule.get("spacing", []) is None:
        # workaround. I dislike this solution.
        config.datamodule.spacing = data_module.preprocess_info["spacing"]

    # ------------
    # model
    # ------------
    logger.info("Instantiating model")
    if isinstance(data_module, QualityRegressionDataModule):
        config.csf_image.hparams.output_names = data_module.metric_target_names
        # not used
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
    seg_model = None
    if not twostage:
        seg_model = setup_segmentation_model(config, load_best_ckpt=False)
    model = setup_image_csf_model(config, seg_model)
    # ------------
    # testing
    # ------------
    # Init callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.test.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf['_target_']}>")
                if (
                    cb_conf["_target_"].split(".")[-1]
                    in ["PredictionWriter", "PixelConfidenceWriter", "MultiPredictionWriter"]
                    and twostage
                ):
                    # Prediction masks are already saved by stage 1.
                    continue
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

    logger.info("Starting testing...")
    trainer.test(model, datamodule=data_module)

    # ------------
    # evaluation
    # ------------
    expt_dir = Path(config.paths.output_dir)
    evaluate_failures(
        ExperimentData.from_experiment_dir(expt_dir),
        output_dir=Path(config.paths.analysis_dir),
        config=config.analysis,
    )
    if config.analysis.get("ood_metrics", False):
        evaluate_ood(
            ExperimentData.from_experiment_dir(expt_dir),
            output_dir=Path(config.paths.analysis_dir),
            config=config.analysis,
        )

    # Save configuration diff at the end of testing to capture any runtime changes
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
