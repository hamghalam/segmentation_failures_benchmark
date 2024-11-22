from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

import segmentation_failures.evaluation.ood_detection.metrics as ood_metrics
from segmentation_failures.evaluation.experiment_data import ExperimentData


def compute_ood_scores(
    confid_arr: np.ndarray,
    ood_labels: np.ndarray,
    query_metrics: List[str],
) -> Dict[str, float]:
    """Based on confidence values and OOD labels, compute some OOD detection metrics.

    Args:
        confid_arr (np.ndarray): 1D-Array with confidences.
        ood_labels (np.ndarray): 1D-Array with OOD-labels (binary).
        query_metrics (List[str]): List of OOD metrics to compute for the data.

    Returns:
        Dict[str, float]: Dictionary of scalar OOD metrics.
    """
    # arrays should be 1D and have the same length
    assert len(confid_arr.shape) == len(ood_labels.shape) == 1
    assert len(confid_arr) == len(ood_labels)
    ood_scores: Dict[str, float] = {}
    if np.any(np.isnan(confid_arr)):
        logger.warning("NaN values in confidence scores. Inserting NaN in metrics.")
        for score in query_metrics:
            ood_scores[score] = np.nan
        return ood_scores
    stats = ood_metrics.StatsCache(
        scores=-confid_arr,  # higher confidence -> lower ood score
        ood_labels=ood_labels,
    )
    # scores
    for score in query_metrics:
        score_fn = ood_metrics.get_metric_function(score)
        ood_scores[score] = score_fn(stats)
    # curves: maybe later
    return ood_scores


def evaluate_ood(expt_data: ExperimentData, output_dir: Path, config: OmegaConf):
    # this should compute different FD-metrics and save them as a dataframe to the output_dir
    # I just compute one risk for every segmentation metric present in the dataframe.
    id_domains = config.id_domain
    if id_domains is None:
        logger.warning("No ID domain specified. Skipping OOD analysis.")
        return
    if isinstance(id_domains, str):
        id_domains = [id_domains]
    domains = np.unique(expt_data.domain_names).tolist()
    if set(domains) == set(id_domains):
        logger.warning("All domains are ID domains. Skipping OOD analysis.")
        return
    domains.append("all_ood_")  # also evaluate on all ood domains together
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "ood_metrics.csv"
    if output_file.exists():
        # get an alternative file name
        i = 1
        while output_file.exists():
            output_file = output_dir / f"ood_metrics_{i}.csv"
            i += 1
        logger.warning(
            f"Output file {output_dir / 'ood_metrics.csv'} already exists. Saving to {output_file} instead."
        )

    # Also compute OOD scores
    analysis_results = []
    ood_metrics = config.ood_metrics
    if len(ood_metrics) == 0:
        return
    if not set(id_domains).issubset(domains):
        logger.warning(
            f"ID domain(s) {id_domains} not found in experiment data. Maybe it is misconfigured?"
        )
    for curr_domain in domains:
        if curr_domain in id_domains:
            continue
        for confid_idx, confid_name in enumerate(expt_data.confid_scores_names):
            id_mask = np.isin(np.array(expt_data.domain_names), id_domains)
            if curr_domain == "all_ood_":
                ood_mask = np.logical_not(id_mask)
            else:
                ood_mask = np.array(expt_data.domain_names) == curr_domain
            testset_mask = np.logical_or(ood_mask, id_mask)
            subset_confid = expt_data.confid_scores[testset_mask, confid_idx]
            subset_labels = ood_mask[testset_mask].astype(int)

            scores = compute_ood_scores(
                confid_arr=subset_confid,
                ood_labels=subset_labels,
                query_metrics=ood_metrics,
            )
            result_row = {
                "confid_name": confid_name,
                "domain": curr_domain,
                "n_cases_id": np.sum(id_mask),
                "n_cases_ood": np.sum(ood_mask),
            }
            result_row.update(scores)
            analysis_results.append(result_row)

    pd.DataFrame(analysis_results).to_csv(output_file)
