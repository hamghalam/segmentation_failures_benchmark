import numpy as np

from segmentation_failures.evaluation.ood_detection.metrics import (
    StatsCache,
    get_metric_function,
)


def test_auroc():
    # test this with classification
    np.random.seed(42)
    pseudo_ood_labels = np.random.randint(0, 2, size=100000)
    rand_confids = np.random.rand(100000)
    perfect_confids = 1 - pseudo_ood_labels
    auroc = get_metric_function("ood_auc")
    result1 = auroc(StatsCache(scores=-rand_confids, ood_labels=pseudo_ood_labels))
    result2 = auroc(StatsCache(scores=-perfect_confids, ood_labels=pseudo_ood_labels))
    # Since the true_eaurc is only an approximation, the results are not exactly equal
    assert np.isclose(result1, 0.5, atol=1e-2)
    assert result2 == 1.0
