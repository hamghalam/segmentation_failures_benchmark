import torch
from loguru import logger
from pytorch_lightning import LightningModule

from segmentation_failures.models.confidence_aggregation.base import (
    AbstractAggregator,
    AbstractEnsembleAggregator,
)
from segmentation_failures.utils.data import load_dataset_json
from segmentation_failures.utils.label_handling import convert_to_onehot_batch


class SimpleAggModule(LightningModule):
    def __init__(
        self,
        aggregation_methods: dict[str, AbstractAggregator],
        dataset_id: int,
    ) -> None:
        super().__init__()
        if not hasattr(aggregation_methods, "items"):
            raise ValueError("Aggregation methods must be a dict.")
        self.aggregators = {
            k: am for k, am in aggregation_methods.items() if isinstance(am, AbstractAggregator)
        }
        self.ensemble_aggregators = {
            k: am
            for k, am in aggregation_methods.items()
            if isinstance(am, AbstractEnsembleAggregator)
        }
        dataset_json = load_dataset_json(dataset_id)
        self.regions_or_labels = [v for _, v in dataset_json["labels"].items()]

    def test_step(self, batch, batch_idx):
        # assume batch keys ["confid", "pred", "pred_samples", "confid_names"]
        # predictions are label maps here, not logits!
        aggregated_confid = {}
        assert batch["pred"].shape[1] == 1, batch["pred"].shape
        prediction = batch["pred"].squeeze(1)  # labels, shape BHW[D]
        prediction_distr = batch.get("pred_samples", None)  # shape BKHW[D], K = samples
        for confid_idx, pxl_confid in enumerate(batch["confid_names"]):
            if isinstance(pxl_confid, (list, tuple)):
                # artefact of default collation
                assert len(pxl_confid) == 1
                pxl_confid = pxl_confid[0]
            # confid shape BCHW[D]
            confid_map = batch["confid"][:, confid_idx]
            # NOTE prediction should be identical in each iteration
            for agg_name, agg_fn in self.aggregators.items():
                confid_name = f"{pxl_confid}_{agg_name}"
                aggregated_confid[confid_name] = agg_fn(prediction, confid_map).cpu()
        if len(self.ensemble_aggregators) > 0 and prediction_distr is None:
            logger.warning(
                f"The aggregations {list(self.ensemble_aggregators)} were configured, but no ensemble predictions are available."
            )
        elif len(self.ensemble_aggregators) > 0 and prediction_distr.shape[1] > 1:
            # these aggregators need a one/multi-hot label map
            n = prediction_distr.shape[1]
            batch_size = prediction_distr.shape[0]
            # Note: shape KBCHW[D]
            onehot_lab = torch.zeros(
                (n, batch_size, len(self.regions_or_labels)) + prediction.shape[1:]
            )
            for b in range(batch_size):
                onehot_lab[:, b] = convert_to_onehot_batch(
                    prediction_distr[b].unsqueeze(1), self.regions_or_labels
                )
            onehot_lab = onehot_lab.to(torch.bool)
            # can also be multihot
            for agg_name, agg_fn in self.ensemble_aggregators.items():
                aggregated_confid[agg_name] = agg_fn(onehot_lab).cpu()
        out_dict = {
            "prediction": prediction,
            "confidence": aggregated_confid,
            "prediction_distr": prediction_distr,  # this might cause memory problems
        }
        return out_dict

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("This confidence score is not trained.")
