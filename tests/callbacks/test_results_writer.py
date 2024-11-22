from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO

from segmentation_failures.evaluation.segmentation.compute_seg_metrics import (
    compute_metrics_for_file,
)
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics,
)


def test_compute_metrics_for_file(tmp_path: Path):
    NUM_LABELS = 3
    BATCH_SIZE = 1
    region_based = False
    metric_multi_fn = get_metrics(["dice"])
    # metric_single_fn = get_metrics(["generalized_dice"])
    pred_dir = tmp_path / "preds"
    lab_dir = tmp_path / "labels"
    pred_dir.mkdir()
    lab_dir.mkdir()
    if region_based:
        all_labels = [0]
        for i in range(1, NUM_LABELS):
            all_labels.insert(1, tuple(range(i, NUM_LABELS)))
    else:
        all_labels = list(range(NUM_LABELS))
    # save dummy label and prediction for evaluation
    preds = torch.randn(BATCH_SIZE, NUM_LABELS, 3, 3, 1)  # logits
    targets = torch.argmax(preds, dim=1).to(torch.uint8)
    outputs = {
        "confidence": torch.zeros(len(preds)),
        "prediction": preds,
    }
    batch = {
        "target": targets,
        "keys": [f"case_{idx:03d}" for idx in range(BATCH_SIZE)],
        "properties": [{"spacing": [1, 1, 1]}] * BATCH_SIZE,
    }
    if region_based:
        raise NotImplementedError
    else:
        class_prediction = torch.argmax(outputs["prediction"], dim=1).numpy().astype(np.uint8)
    reader_fn = NibabelIO().read_seg
    for i, case_id in enumerate(batch["keys"]):
        # I just save predictions here like this because the callback needs too much nnunet stuff.
        label_file = lab_dir / f"{case_id}.nii.gz"
        pred_file = pred_dir / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(class_prediction[i], np.eye(4)), pred_file)
        nib.save(
            nib.Nifti1Image(batch["target"][i].numpy(), np.eye(4)),
            label_file,
        )
        metrics, metrics_multi = compute_metrics_for_file(
            metric_multi_fn, label_file, pred_file, all_labels, seg_reader_fn=reader_fn
        )
        assert all([np.size(x) == 1 for x in metrics.values()])
        if region_based:
            assert all([np.size(x) == len(all_labels) for x in metrics_multi.values()])
        else:
            # no background class
            assert all([np.size(x) == (len(all_labels) - 1) for x in metrics_multi.values()])
