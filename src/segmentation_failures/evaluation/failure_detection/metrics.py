# Adapted from https://github.com/IML-DKFZ/fd-shifts/tree/main
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
import scipy
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


def compute_optimal_aurc(risks):
    tmp_stat = StatsCache(confids=-risks, risks=risks)
    return aurc(tmp_stat)


def compute_random_aurc(risks: np.ndarray):
    return np.mean(risks)


@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values associated with the predictions
        risks (array_like): Risk values associated with the predictions
    """

    confids: npt.NDArray[Any]
    risks: npt.NDArray[Any]

    def check_if_risks_binary(self):
        if set(self.risks) != {0, 1}:
            raise ValueError("Risks are not binary.")

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        self.check_if_risks_binary()
        fpr, tpr, _ = skm.roc_curve(self.risks, self.confids, pos_label=0)
        return fpr, tpr

    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        selective_risks = []
        assert (
            len(self.risks.shape) == 1
            and len(self.confids.shape) == 1
            and len(self.risks) == len(self.confids)
        )

        n_samples = len(self.risks)
        idx_sorted = np.argsort(self.confids)

        coverage = n_samples
        error_sum = sum(self.risks[idx_sorted])

        coverages.append(coverage / n_samples)
        selective_risks.append(error_sum / n_samples)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.risks[idx_sorted[i]]
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_samples)
                selective_risks.append(error_sum / (n_samples - 1 - i))
                weights.append(tmp_weight / n_samples)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            selective_risks.append(selective_risks[-1])
            weights.append(tmp_weight / n_samples)

        return coverages, selective_risks, weights


def register_metric_func(name: str) -> Callable:
    def _inner_wrapper(func: Callable) -> Callable:
        _metric_funcs[name] = func
        return func

    return _inner_wrapper


def get_metric_function(metric_name: str) -> Callable[[StatsCache], float]:
    return _metric_funcs[metric_name]


@register_metric_func("failauc")
@register_metric_func("ood_auc")
@may_raise_sklearn_exception
def failauc(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)


@register_metric_func("fpr@95tpr")
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


@register_metric_func("failap_suc")
@may_raise_sklearn_exception
def failap_suc(stats_cache: StatsCache) -> float:
    stats_cache.check_if_risks_binary()
    return cast(
        float,
        skm.average_precision_score(stats_cache.risks, stats_cache.confids, pos_label=0),
    )


@register_metric_func("failap_err")
@may_raise_sklearn_exception
def failap_err(stats_cache: StatsCache):
    stats_cache.check_if_risks_binary()
    return cast(
        float,
        skm.average_precision_score(stats_cache.risks, -stats_cache.confids, pos_label=1),
    )


@register_metric_func("aurc")
@may_raise_sklearn_exception
def aurc(stats_cache: StatsCache):
    _, risks, weights = stats_cache.rc_curve_stats
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])


@register_metric_func("e-aurc")
@may_raise_sklearn_exception
def eaurc(stats_cache: StatsCache):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    aurc_opt = compute_optimal_aurc(stats_cache.risks)
    return aurc(stats_cache) - aurc_opt


@register_metric_func("opt-aurc")
@may_raise_sklearn_exception
def optimal_aurc(stats_cache: StatsCache):
    return compute_optimal_aurc(stats_cache.risks)


@register_metric_func("rand-aurc")
@may_raise_sklearn_exception
def random_aurc(stats_cache: StatsCache):
    return compute_random_aurc(stats_cache.risks)


@register_metric_func("norm-aurc")
@may_raise_sklearn_exception
def normalized_aurc(stats_cache: StatsCache):
    """Compute actually normalized AURC, i.e. normalize to range between optimal (1) and random CSF (0)."""
    aurc_opt = compute_optimal_aurc(stats_cache.risks)
    aurc_rand = compute_random_aurc(stats_cache.risks)
    return (aurc_rand - aurc(stats_cache)) / (aurc_rand - aurc_opt)


@register_metric_func("spearman")
@may_raise_sklearn_exception
def spearman_correlation(stats_cache: StatsCache):
    return scipy.stats.spearmanr(stats_cache.confids, stats_cache.risks).statistic


@register_metric_func("pearson")
@may_raise_sklearn_exception
def pearson_correlation(stats_cache: StatsCache):
    return scipy.stats.pearsonr(stats_cache.confids, stats_cache.risks).statistic
