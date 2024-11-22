from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

import segmentation_failures.evaluation.failure_detection.metrics as segfail_metrics
from segmentation_failures.evaluation.experiment_data import ExperimentData
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    MetricsInfo,
    get_metrics_and_info,
)


def compute_fd_scores(
    confid_arr: np.ndarray,
    metric_arr: np.ndarray,
    metric_info: MetricsInfo,
    query_fd_metrics: List[str],
    failure_thresh: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    # arrays should be 1D and have the same length
    assert len(confid_arr.shape) == len(metric_arr.shape) == 1
    assert len(confid_arr) == len(metric_arr)
    fd_scores: dict[str, float] = {}
    fd_curves: dict[str, dict[str, np.ndarray]] = {}  # 2D array

    if np.any(np.isnan(confid_arr)) or np.any(np.isnan(metric_arr)):
        logger.warning(
            "NaN values in confidence scores or segmentation metrics. Inserting NaN in FD-metrics."
        )
        for score in query_fd_metrics:
            fd_scores[score] = np.nan
        return fd_scores, fd_curves

    # Compute risk based on metric_arr and metric_info
    risk_arr = metric_arr.copy()
    if failure_thresh is not None:
        # binary risk
        if metric_info.higher_better:
            risk_arr = metric_arr < failure_thresh
        else:
            risk_arr = metric_arr > failure_thresh
    elif metric_info.higher_better:
        # continuous risk
        risk_arr *= -1
        if metric_info.max_value < np.inf:
            risk_arr += metric_info.max_value

    stats = segfail_metrics.StatsCache(
        confids=confid_arr,
        risks=risk_arr,
    )
    # scores
    for score in query_fd_metrics:
        fd_fn = segfail_metrics.get_metric_function(score)
        fd_scores[score] = fd_fn(stats)

    # curves
    coverages, selective_risks, weights = stats.rc_curve_stats
    fd_curves.update(
        {
            "risk_coverage_curve": {
                "coverage": coverages,
                "risk": selective_risks,
                "weight": weights,
            },
        }
    )
    return fd_scores, fd_curves


def check_analysis_config(config: OmegaConf):
    expected_keys = ["id_domain", "save_curves", "fd_metrics", "fail_thresholds"]
    missing_keys = []
    for k in expected_keys:
        if k not in expected_keys:
            missing_keys.append(k)
    if len(missing_keys) > 0:
        raise KeyError(f"Could not find key {k} in analysis configuration: {config}")


def evaluate_failures(expt_data: ExperimentData, output_dir: Path, config: OmegaConf):
    # this should compute different FD-metrics and save them as a dataframe to the output_dir
    # I just compute one risk for every segmentation metric present in the dataframe.
    _, all_metric_infos = get_metrics_and_info()
    id_domains = config.id_domain
    domains = np.unique(expt_data.domain_names).tolist()
    domains.append("all_ood_")  # also evaluate on all ood domains together
    domains.append("all_")  # also evaluate on all domains together
    if not set(id_domains).issubset(domains):
        logger.warning(
            f"ID domain(s) {id_domains} not found in experiment data. Maybe it is misconfigured?"
        )
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "fd_metrics.csv"
    if output_file.exists():
        # get an alternative file name
        i = 1
        while output_file.exists():
            output_file = output_dir / f"fd_metrics_{i}.csv"
            i += 1
        logger.warning(
            f"Output file {output_dir / 'fd_metrics.csv'} already exists. Saving to {output_file} instead."
        )

    check_analysis_config(config)
    fail_thresholds = config.fail_thresholds
    if fail_thresholds is None:
        fail_thresholds = dict()
    analysis_results = []
    fd_metrics = config.fd_metrics
    for curr_domain in domains:
        for confid_idx, confid_name in enumerate(expt_data.confid_scores_names):
            for metric_idx, metric_name in enumerate(expt_data.segmentation_metrics_names):
                if curr_domain == "all_ood_":
                    domain_mask = np.logical_not(
                        np.isin(np.array(expt_data.domain_names), id_domains)
                    )
                    if domain_mask.sum() == 0:
                        logger.warning(
                            "No OOD domains found in the data. Skipping all_ood_ FD evaluation."
                        )
                        continue
                elif curr_domain == "all_":
                    domain_mask = np.ones_like(expt_data.domain_names, dtype=bool)
                else:
                    domain_mask = np.array(expt_data.domain_names) == curr_domain
                # I remove the mean_ because I compute these automatically and don't have a separate info
                metric_info = all_metric_infos[metric_name.removeprefix("mean_")]
                thresh = fail_thresholds.get(metric_name.removeprefix("mean_"), None)
                # For each confidence score, compute FD metrics based on each segmentation metric
                scores, curves = compute_fd_scores(
                    confid_arr=expt_data.confid_scores[domain_mask, confid_idx],
                    metric_arr=expt_data.segmentation_metrics[domain_mask, metric_idx],
                    metric_info=metric_info,
                    query_fd_metrics=fd_metrics,
                    failure_thresh=thresh,
                )
                result_row = {
                    "confid_name": confid_name,
                    "metric": metric_name,
                    "domain": curr_domain,
                    "n_cases": np.sum(domain_mask),
                }
                result_row.update(scores)
                if config.save_curves:
                    # save curves; for now, just a npz file with the name of the curve in the output_dir.
                    for curve_name, curve in curves.items():
                        # curve must be a dict with numpy arrays as values
                        curves_dir = output_dir / curve_name
                        curves_dir.mkdir(exist_ok=True)
                        file_name = f"{curr_domain}_{confid_name}_{metric_name}.npz"
                        result_row[f"file_{curve_name}"] = str(
                            curves_dir.relative_to(output_dir) / file_name
                        )
                        np.savez(curves_dir / file_name, **curve)
                analysis_results.append(result_row)

    pd.DataFrame(analysis_results).to_csv(output_file)
