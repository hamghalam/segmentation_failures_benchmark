# Adapted from https://github.com/IML-DKFZ/fd-shifts/tree/main
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
from sklearn import metrics as skm
from typing_extensions import ParamSpec

_metric_funcs = {}

T = TypeVar("T")
P = ParamSpec("P")


def may_raise_sklearn_exception(func: Callable[P, T]) -> Callable[P, T]:
    def _inner_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ValueError:
            return cast(T, np.nan)

    return _inner_wrapper


@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values associated with the predictions
        risks (array_like): Risk values associated with the predictions
    """

    scores: npt.NDArray[Any]
    ood_labels: npt.NDArray[Any]

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        fpr, tpr, _ = skm.roc_curve(self.ood_labels, self.scores, pos_label=1)
        return fpr, tpr

    # maybe add PR curve stats here


def register_metric_func(name: str) -> Callable:
    def _inner_wrapper(func: Callable) -> Callable:
        _metric_funcs[name] = func
        return func

    return _inner_wrapper


def get_metric_function(metric_name: str) -> Callable[[StatsCache], float]:
    return _metric_funcs[metric_name]


@register_metric_func("ood_auc")
@may_raise_sklearn_exception
def failauc(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)


@register_metric_func("ood_fpr@95tpr")
@may_raise_sklearn_exception
def fpr_at_95_tpr(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    return np.min(fpr[np.argwhere(tpr >= 0.9495)])


@register_metric_func("ood_detection_error@95tpr")
@may_raise_sklearn_exception
def deterror_at_95_tpr(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    tpr_mask = np.argwhere(tpr >= 0.9495)
    fpr95 = np.min(fpr[tpr_mask])
    tpr95 = np.min(tpr[tpr_mask])
    return 0.5 * (1 - tpr95 + fpr95)
