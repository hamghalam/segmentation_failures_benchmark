# from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import monai.metrics as mn_metrics
import numpy as np

from segmentation_failures.evaluation.segmentation.custom_metrics.surface_distance import (
    HausdorffDistanceEmptyHandlingMetric,
    SurfaceDiceEmptyHandlingMetric,
)


@dataclass
class MetricsInfo:
    higher_better: bool = True
    min_value: float = 0.0
    max_value: float = 0.0
    classwise: bool = True


_metric_factories = {}


def register_metric(name: str) -> Callable:
    def _inner_wrapper(metric_factory: Callable) -> Callable:
        _metric_factories[name] = metric_factory
        return metric_factory

    return _inner_wrapper


def get_metrics(
    metric_list: list | str | None = None, **metric_kwargs
) -> Dict[str, mn_metrics.CumulativeIterationMetric]:
    return get_metrics_and_info(metric_list, **metric_kwargs)[0]


def get_metrics_info(
    metric_list: list | str | None = None, **metric_kwargs
) -> Dict[str, MetricsInfo]:
    return get_metrics_and_info(metric_list, **metric_kwargs)[1]


def get_metrics_and_info(
    metric_list: list | str | None = None, **metric_kwargs
) -> Tuple[dict[str, mn_metrics.CumulativeIterationMetric], dict[str, MetricsInfo]]:
    if isinstance(metric_list, str):
        metric_list = [metric_list]
    if metric_list is None:
        # default: get all
        metric_list = list(_metric_factories.keys())
    metrics = {}
    infos = {}
    for k in metric_list:
        metrics[k], infos[k] = _metric_factories[k](**metric_kwargs)
    return metrics, infos


@register_metric("dice")
def dice_score(include_background=False):
    # ignore_empty=False makes sure that cases with empty GT and pred receive a score of 1
    metric = mn_metrics.DiceMetric(
        include_background=include_background, reduction="none", ignore_empty=False
    )
    info = MetricsInfo(True, 0, 1, True)
    return metric, info


@register_metric("hausdorff95")
def hd95_score(include_background=False):
    metric = HausdorffDistanceEmptyHandlingMetric(
        include_background=include_background, percentile=95, reduction="none"
    )
    info = MetricsInfo(False, 0, np.inf, True)
    return metric, info


@register_metric("generalized_dice")
def gen_dice_score(include_background=False, weight_type="square"):
    metric = mn_metrics.GeneralizedDiceScore(
        include_background=include_background,
        reduction="none",
        weight_type=weight_type,
    )
    info = MetricsInfo(True, 0, 1, False)
    return metric, info


@register_metric("surface_dice")
def surface_dice_score(include_background=False, class_thresholds=1.0):
    if not isinstance(class_thresholds, (list, tuple)):
        class_thresholds = [class_thresholds]
    metric = SurfaceDiceEmptyHandlingMetric(
        class_thresholds=class_thresholds,
        include_background=include_background,
        reduction="none",
        use_subvoxels=True,
    )
    info = MetricsInfo(True, 0, 1, True)
    return metric, info
