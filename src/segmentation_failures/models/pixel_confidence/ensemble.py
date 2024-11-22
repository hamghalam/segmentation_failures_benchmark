from typing import List

import pytorch_lightning as pl
import torch
from loguru import logger

from segmentation_failures.models.pixel_confidence.posthoc import (
    compute_confidence_map,
    compute_mean_prediction,
)
from segmentation_failures.models.pixel_confidence.scores import get_pixel_csf


class DeepEnsembleMultiConfidenceSegmenter(pl.LightningModule):
    def __init__(
        self,
        segmentation_net: List[torch.nn.Module],
        csf_names: str | list[str],
        overlapping_classes: bool = False,
        everything_on_gpu=False,
        confidence_precision="32",
        num_models: int = 5,
    ) -> None:
        """NOTE: segmentation net is not a single network but a list!

        It's just named this way for compatibility with the testing pipeline conventions."""
        super().__init__()
        if isinstance(csf_names, str):
            csf_names = [csf_names]
        self.csf_dict = {}
        for csf in csf_names:
            self.csf_dict[csf] = get_pixel_csf(csf)
        self.model_list = torch.nn.ModuleList(segmentation_net)
        if num_models <= 1:
            raise ValueError("Number of models in ensemble must be greater than 1")
        if len(segmentation_net) < num_models:
            raise ValueError("Number of models in ensemble is less than specified")
        if len(segmentation_net) > num_models:
            logger.info(
                f"Number of models in ensemble is greater than specified ({len(segmentation_net)}). "
                f"Using only first {num_models}."
            )
            self.model_list = self.model_list[:num_models]
        self.overlapping_classes = overlapping_classes
        self.everything_on_gpu = everything_on_gpu
        if confidence_precision == "32":
            self.confid_dtype = torch.float32
        elif confidence_precision == "64":
            self.confid_dtype = torch.float64
        else:
            raise ValueError(f"Unknown precision {confidence_precision}")

    def forward(self, x: torch.Tensor, query_confids=None):
        if query_confids is None:
            query_confids = self.csf_dict.keys()
        if isinstance(query_confids, str):
            query_confids = [query_confids]
        logger.debug(
            f"Starting inference of segmentation model ({len(self.model_list)} in ensemble)"
        )
        device = self.device if self.everything_on_gpu else torch.device("cpu")
        logits_distr = self.segmentation_inference(x).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits_distr = logits_distr.to(dtype=self.confid_dtype)
            # prediction shape KBCHW[D], K=#models in ensemble
            logits = compute_mean_prediction(logits_distr, self.overlapping_classes, mc_dim=0)
            for curr_name in query_confids:
                logger.debug(f"Computing confidence map {curr_name}")
                csf_fn = self.csf_dict[curr_name]
                confid = compute_confidence_map(
                    logits_distr, csf_fn, self.overlapping_classes, mc_dim=0
                )
                # generator is used for memory saving, but it probably breaks backpropagation (which I don't need)
                yield {
                    "csf": curr_name,
                    "logits": logits,
                    "confid": confid,
                    "logits_distr": logits_distr,
                }

    def segmentation_inference(self, x):
        # assume x shape BCHW[D]
        device = self.device if self.everything_on_gpu else "cpu"
        logits_distr = torch.zeros(
            (
                len(self.model_list),
                x.shape[0],
                self.model_list[0].hparams.num_classes,
                *x.shape[2:],
            ),
            device=device,
        )
        for i, model in enumerate(self.model_list):
            logits_distr[i] = model(x).to(device=device)
        return logits_distr

    def test_step(self, batch, batch_idx):
        logger.debug(f"Current case ID: {batch['keys']}")
        confid_dict = {}
        confid_map_generator = self(batch["data"])
        for output in confid_map_generator:
            # prediction doesn't change between iterations
            prediction, confid = output["logits"], output["confid"]
            prediction_distr = output["logits_distr"]  # this might cause memory problems
            confid_dict[output["csf"]] = confid
        return {
            "prediction": prediction,
            "confidence_pixel": confid_dict,
            "prediction_distr": prediction_distr,
        }

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("This module is not trained.")
