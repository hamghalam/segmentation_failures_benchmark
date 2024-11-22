"""How to test?
- Test that rejector model works as expected: Generate dummy data (arbitrary regression task), run fit and predict with pipeline
- Test extract_features
"""

import time

import pytest
import torch

from segmentation_failures.models.confidence_aggregation import (
    ForegroundAggregator,
    ForegroundSizeAggregator,
    HeuristicAggregationModule,
    RadiomicsAggregationModule,
)
from segmentation_failures.models.confidence_aggregation.base import (
    PairwiseDiceAggregator,
)


# TODO this fails because one environment variable isn't set
def test_extract_features():
    dummy_module = HeuristicAggregationModule(
        regression_model="regression_forest",
        dataset_id=500,
        confid_name="dummy_confid",
        target_metrics=["generalized_dice"],
        heuristic_list=[ForegroundAggregator(), ForegroundSizeAggregator()],
    )
    # region based is a bit hard to simulate here
    dummy_prediction = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
    )
    dummy_prediction = dummy_prediction.reshape(1, 1, *dummy_prediction.shape)
    dummy_confid = torch.rand_like(dummy_prediction[:, 0], dtype=float)
    dummy_image = torch.rand_like(dummy_prediction, dtype=float)
    features = dummy_module.extract_features(dummy_image, dummy_prediction, dummy_confid)
    assert features.shape == (len(dummy_prediction), len(dummy_module.aggregator_list))


# TODO outdated test
# radiomics requires a trainer mock, which I don't want to implement.
@pytest.mark.parametrize(
    "method",
    [
        "heuristic",
    ],
)
@pytest.mark.parametrize("img_dim", [2, 3])
def test_multiclass_aggregation(method: str, img_dim: int):
    NUM_BATCH = 2
    NUM_CLASSES = 4
    IMG_SIZE = 20
    IMG_SHAPE = [IMG_SIZE] * img_dim

    class SimulateModel:
        def eval(self):
            pass

        def requires_grad_(self, val):
            pass

        def __call__(self, x, confid_name):
            dummy_prediction = torch.randn(size=(NUM_BATCH, NUM_CLASSES, *IMG_SHAPE))
            dummy_confid = torch.rand(NUM_BATCH, *IMG_SHAPE)
            yield {"logits": dummy_prediction, "confid": dummy_confid}

        def forward(self, batch):
            return self(batch)

    if method == "heuristic":
        dummy_module = HeuristicAggregationModule(
            SimulateModel(),
            num_classes=NUM_CLASSES,
            target_metric="generalized_dice",
            confid_name="dummy_confid",
        )
    elif method == "radiomics":
        dummy_module = RadiomicsAggregationModule(
            image_dim=img_dim,
            pixel_csf=SimulateModel(),
            num_classes=NUM_CLASSES,
            target_metric="generalized_dice",
            confid_threshold=0.7,
            confid_name="dummy_confid",
        )
    else:
        raise ValueError
    outputs = []
    for i in range(3):
        batch = {
            "data": torch.rand(NUM_BATCH, 1, *IMG_SHAPE),  # 1 is modality
            "target": torch.randint(NUM_CLASSES, size=(NUM_BATCH, 1, *IMG_SHAPE)),
        }
        outputs.append(dummy_module.training_step(batch, i))
        assert outputs[-1]["quality_true"].shape == (NUM_BATCH,)
    dummy_module.on_train_epoch_end()


def test_pairwise_dice_agg(num_batch=4, img_size=(5, 5), region_based=True):
    NUM_CLASSES = 2
    NUM_SAMPLES = 4
    consensus_pred = torch.zeros(NUM_SAMPLES, num_batch, NUM_CLASSES, *img_size)
    start_x = 1
    start_y = 1
    size = 2
    consensus_pred[:, :, 1, start_x : start_x + size, start_y : start_y + size] = 1
    if region_based:
        consensus_pred[:, :, 0, 0:size, -size:] = 1
    consensus_pred = consensus_pred.to(dtype=torch.bool)
    score = PairwiseDiceAggregator(include_zero_label=region_based)
    start = time.time()
    result = score.aggregate(consensus_pred)
    end = time.time()
    print(f"Time taken for pairwise dice: {end - start} seconds")
    assert torch.allclose(result, torch.ones_like(result))

    disjoint_pred = torch.zeros(NUM_SAMPLES, num_batch, NUM_CLASSES, *img_size)
    for i in range(len(disjoint_pred)):
        disjoint_pred[i, :, 1, i] = 1
        disjoint_pred[i, :, 0] = 1 - disjoint_pred[i, :, 1]
        if region_based:
            disjoint_pred[i, :, 0] = disjoint_pred[i, :, 1]

    disjoint_pred = disjoint_pred.to(dtype=torch.bool)
    result = score.aggregate(disjoint_pred)
    assert torch.allclose(result, torch.zeros_like(result))
