import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from segmentation_failures.evaluation import ExperimentData, evaluate_failures
from segmentation_failures.evaluation.ood_detection.ood_analysis import evaluate_ood
from segmentation_failures.evaluation.segmentation.compute_seg_metrics import (
    compute_metrics_for_prediction_dir,
)
from segmentation_failures.utils.data import get_dataset_dir, load_dataset_json


def cli_main():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    expt_path = Path(args.expt_path)
    logger.info(f"Starting evaluation for experiment: {expt_path}")

    if not (expt_path / ".hydra/config.yaml").exists():
        raise FileNotFoundError(f"Could not find hydra configuration in {expt_path}. Exiting...")
    expt_data = ExperimentData.from_experiment_dir(expt_path)
    expt_config = expt_data.config  # this is a DictConfig

    # need to have an analysis configuration
    if args.analysis_config:
        # In my current analysis configuration, there is a defaults list, which is resolved by hydra
        from hydra import compose, initialize_config_dir

        ana_cfg_path = Path(args.analysis_config)
        initialize_config_dir(
            config_dir=str(ana_cfg_path.parent), version_base=None, job_name="dummy"
        )
        cfg = compose(config_name=str(ana_cfg_path.name))
        # Need to update the experiment configuration here because
        # my analysis configuration contains a reference to it (id_domain)
        expt_config.analysis = cfg
    analysis_config = expt_config.analysis

    analysis_dir = args.out_path
    if analysis_dir is None:
        analysis_dir = expt_data.config.paths.analysis_dir
        backup_previous_runs(analysis_dir)
    analysis_dir = Path(analysis_dir)

    if args.add_mean_quality_regression:
        if expt_config.expt_name.split("-")[-1] not in [
            "quality_regression",
            "predictive_entropy+heuristic",
            "predictive_entropy+radiomics",
        ]:
            raise ValueError(
                "Experiment name doesn't support mean confidence! Only quality regression methods are supported."
            )
        # this also saves the updated experiment data
        expt_data = add_mean_confid_quality_regression(
            expt_data, output_dir=expt_data.config.paths.results_dir
        )

    # recompute the segmentation metrics if requested
    if args.recompute_metrics is not None:
        expt_data = recompute_metrics(
            args.recompute_metrics,
            expt_data,
            output_dir=expt_data.config.paths.results_dir,
            config=expt_data.config,
            n_proc=args.nproc,
        )

    if args.remove_lowmedium and expt_data.config.dataset.dataset_id == "500":
        # remove low and medium shift domain cases from the analysis
        keep_indices = [
            i
            for i, d in enumerate(expt_data.domain_names)
            if not (d.endswith("-low") or d.endswith("-medium"))
        ]
        expt_data.confid_scores = expt_data.confid_scores[keep_indices]
        expt_data.domain_names = [
            d for i, d in enumerate(expt_data.domain_names) if i in keep_indices
        ]
        expt_data.case_ids = [d for i, d in enumerate(expt_data.case_ids) if i in keep_indices]
        expt_data.segmentation_metrics = expt_data.segmentation_metrics[keep_indices]
        expt_data.segmentation_metrics_multi = expt_data.segmentation_metrics_multi[keep_indices]

    if args.remove_runmc and expt_data.config.dataset.dataset_id == "521":
        # remove low and medium shift domain cases from the analysis
        keep_indices = [i for i, d in enumerate(expt_data.domain_names) if d != "RUNMC"]
        expt_data.confid_scores = expt_data.confid_scores[keep_indices]
        expt_data.domain_names = [
            d for i, d in enumerate(expt_data.domain_names) if i in keep_indices
        ]
        expt_data.case_ids = [d for i, d in enumerate(expt_data.case_ids) if i in keep_indices]
        expt_data.segmentation_metrics = expt_data.segmentation_metrics[keep_indices]
        expt_data.segmentation_metrics_multi = expt_data.segmentation_metrics_multi[keep_indices]

    evaluate_failures(expt_data, output_dir=analysis_dir, config=analysis_config)
    if not args.no_ood:
        evaluate_ood(expt_data, output_dir=analysis_dir, config=analysis_config)

    # save the analysis configuration
    analysis_config_file = analysis_dir / "analysis_config.yaml"
    logger.info(f"Saving analysis configuration to {analysis_config_file}")
    OmegaConf.save(analysis_config, analysis_config_file)


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "expt_path",
        type=str,
        help="Path to the experiment run directory upon which the evaluation should be based.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=None,
        help="Path to the directory where outputs are saved.",
    )
    parser.add_argument(
        "--analysis_config",
        type=str,
        default=None,
        help="Path to a configuration file for the analysis. Overwrites corresponding experiment config entries.",
    )
    parser.add_argument(
        "--recompute_metrics",
        type=str,
        nargs="*",
        default=None,
        help="Recompute the segmentation metrics. Otherwise, only the analysis is performed.",
    )
    parser.add_argument(
        "--no_ood",
        action="store_true",
        help="Do not evaluate OOD detection metrics.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes to use for metric computation.",
    )
    parser.add_argument(
        "--remove_lowmedium",
        action="store_true",
        help="Remove low and medium shift domain cases from the analysis (only for Dataset500!).",
    )
    parser.add_argument(
        "--remove_runmc",
        action="store_true",
        help="Remove RUNMC domain cases from the analysis (only for Dataset520!).",
    )
    parser.add_argument(
        "--add_mean_quality_regression",
        action="store_true",
        help="Add a mean confidence score for all class-wise metrics.",
    )


def recompute_metrics(
    metric_list,
    expt_data: ExperimentData,
    output_dir: str = None,
    config: OmegaConf = None,
    n_proc: int = 1,
):
    if config is None:
        config = expt_data.config
    if output_dir is not None:
        # backup old results if output_dir is not empty
        backup_previous_runs(output_dir)
    if len(metric_list) == 0:
        # evaluate same metrics as in the original experiment
        metric_list = (
            expt_data.segmentation_metrics_names + expt_data.segmentation_metrics_names_multi
        )
    # this is a special case hack, because currently mean metrics are computed automatically...
    metric_list = [m for m in metric_list if not m.startswith("mean_")]
    # todo some preparation steps
    # get the dataset directory (assume test set)
    dataset_id = config.dataset.dataset_id
    dataset_dir = get_dataset_dir(dataset_id, config.paths.data_root_dir)
    # get a list of label files
    dataset_json = load_dataset_json(dataset_id, config.paths.data_root_dir)
    suffix = dataset_json.get("file_ending", ".nii.gz")
    label_file_list = [x for x in (dataset_dir / "labelsTs").glob("*" + suffix)]
    case_ids = [x.name.removesuffix(suffix) for x in label_file_list]
    if set(case_ids) != set(expt_data.case_ids):
        raise ValueError(f"Case IDs do not match: {set(case_ids) - set(expt_data.case_ids)}")
    # order label files according to the expt_data case ids
    label_file_list = sorted(
        label_file_list,
        key=lambda x: expt_data.case_ids.index(x.name.removesuffix(suffix)),
    )
    assert [x.name.removesuffix(suffix) for x in label_file_list] == expt_data.case_ids
    logger.info(f"Recomputing segmentation metrics {metric_list} for {len(label_file_list)} cases")
    metrics_dict, multi_metrics_dict = compute_metrics_for_prediction_dir(
        metric_list=metric_list,
        prediction_dir=config.paths.predictions_dir,
        label_file_list=label_file_list,
        dataset_id=dataset_id,
        num_processes=n_proc,
    )
    # dicts with shape (n_samples,) and (n_samples, n_classes), respectively.
    assert all(arr.shape[0] == len(case_ids) for arr in metrics_dict.values())
    if len(multi_metrics_dict) > 0:
        assert all(arr.shape[0] == len(case_ids) for arr in multi_metrics_dict.values())

    # Update experiment data
    expt_data.segmentation_metrics = np.stack([metrics_dict[k] for k in metrics_dict], axis=-1)
    expt_data.segmentation_metrics_names = list(metrics_dict.keys())
    if len(multi_metrics_dict) > 0:
        expt_data.segmentation_metrics_multi = np.stack(
            [multi_metrics_dict[k] for k in multi_metrics_dict], axis=-1
        )
    else:
        expt_data.segmentation_metrics_multi = np.array([])
    expt_data.segmentation_metrics_names_multi = list(multi_metrics_dict.keys())
    if output_dir is not None:
        expt_data.save(output_dir)
    return expt_data


def backup_previous_runs(orig_dir: str):
    if len(list(Path(orig_dir).iterdir())) == 0:
        return
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(orig_dir) / "backup_{}".format(time_str)
    backup_dir.mkdir()
    logger.info(f"Backing up previous run:\n{orig_dir} -> {backup_dir}")
    # Move all files and directories in analysis_dir to backup_dir
    for item in Path(orig_dir).iterdir():
        if not (item.is_dir() and item.name.startswith("backup_")):
            shutil.move(item, backup_dir / item.name)


def add_mean_confid_quality_regression(expt_data, output_dir: str = None):
    # Add a "mean" confidence score for all class-wise metrics
    confid_names = expt_data.confid_scores_names
    multiclass_metric_names = expt_data.segmentation_metrics_names_multi
    num_classes = expt_data.segmentation_metrics_multi.shape[1]
    if num_classes == 1:
        logger.info("No multi-class metrics found. Skipping...")
        return expt_data

    for metric_name in multiclass_metric_names:
        mean_metric_name = f"mean_{metric_name}"
        if mean_metric_name in confid_names:
            logger.info(f"Mean confidence score for {metric_name} already exists. Skipping...")
            continue
        logger.info(f"Adding mean confidence score for {metric_name}")
        # get matching confidence scores
        matching_confids = [x for x in confid_names if x.split("_")[0] == metric_name]
        if len(matching_confids) == 0:
            logger.warning(f"No confidence scores found for {metric_name}. Skipping...")
            continue
        elif len(matching_confids) != num_classes:
            raise ValueError(
                f"Found {len(matching_confids)} confidence scores for {metric_name}. Expected {num_classes}"
            )
        # compute mean confidence score
        mean_confid = expt_data.confid_scores[
            :, [confid_names.index(x) for x in matching_confids]
        ].mean(axis=1)
        expt_data.confid_scores = np.concatenate(
            [expt_data.confid_scores, mean_confid[:, None]], axis=1
        )
        expt_data.confid_scores_names.append(mean_metric_name)

    # Update experiment data
    if output_dir is not None:
        # backup old results if output_dir is not empty
        backup_previous_runs(output_dir)
        expt_data.save(output_dir)
    return expt_data


if __name__ == "__main__":
    cli_main()
