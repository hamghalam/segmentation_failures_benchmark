import time

import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import LightningModule

from segmentation_failures.models.pixel_confidence.scores import (
    PixelConfidenceScore,
    get_pixel_csf,
)
from segmentation_failures.utils.network import disable_dropout, enable_dropout


class PosthocMultiConfidenceSegmenter(LightningModule):
    def __init__(
        self,
        segmentation_net: torch.nn.Module,
        csf_names: str | list[str],
        num_mcd_samples: int = 0,
        overlapping_classes: bool = False,
        everything_on_gpu=False,
        confidence_precision="32",
    ) -> None:
        super().__init__()
        self.model = segmentation_net
        if isinstance(csf_names, str):
            csf_names = [csf_names]
        self.csf_dict = {}
        for csf in csf_names:
            self.csf_dict[csf] = get_pixel_csf(csf)
        self.num_mcd_samples = num_mcd_samples
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
            f"Starting inference of segmentation model ({self.num_mcd_samples} MC-samples)"
        )
        device = self.device if self.everything_on_gpu else torch.device("cpu")
        start_time = time.time()
        if self.num_mcd_samples > 0:
            logits_distr = self.model_mcd(x)
        else:
            logits_distr = self.model(x).unsqueeze(0)
        logger.debug(f"Segmentation model inference took {time.time() - start_time:.2f}s")

        logits_distr = logits_distr.to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits_distr = logits_distr.to(dtype=self.confid_dtype)
            # prediction shape KBCHW[D], K=#MCdropout
            logits = compute_mean_prediction(logits_distr, self.overlapping_classes, mc_dim=0)
            logger.debug("Starting confidence computation")
            for curr_name in query_confids:
                start_time = time.time()
                logger.debug(f"Computing confidence map {curr_name}")
                csf_fn = self.csf_dict[curr_name]
                confid = compute_confidence_map(
                    logits_distr, csf_fn, self.overlapping_classes, mc_dim=0
                )
                logger.debug(f"Computing confidence map took {time.time() - start_time:.2f}s")
                # NOTE generator is used for memory saving, but it probably breaks backpropagation (which I don't need)
                yield {
                    "csf": curr_name,
                    "logits": logits,
                    "confid": confid,
                    "logits_distr": logits_distr,
                }

    def model_mcd(self, x: torch.Tensor):
        # assume x shape BCHW[D]
        device = self.device if self.everything_on_gpu else "cpu"
        enable_dropout(self.model)
        logits = torch.zeros(
            (self.num_mcd_samples, x.shape[0], self.model.hparams.num_classes, *x.shape[2:]),
            device=device,
        )
        for i in range(self.num_mcd_samples):
            # Alternatively, I could "freeze" the dropout for each sliding window inference
            logits[i] = self.model(x).to(device=device)
        disable_dropout(self.model)
        return logits

    def test_step(self, batch, batch_idx):
        logger.debug(f"Current case ID: {batch['keys']}")
        confid_dict = {}
        confid_map_generator = self(batch["data"])
        for output in confid_map_generator:
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
        raise NotImplementedError("This confidence score is not trained.")


def compute_mean_prediction(
    logits: torch.Tensor,
    overlapping_classes: bool,
    mc_dim: int,
):
    class_dim = 1 + 1 * (mc_dim < 2)
    if logits.shape[mc_dim] <= 1:
        return logits.squeeze(mc_dim)
    # Compute mean prediction over MC samples
    # NOTE this just averages the logits, the caller is responsible for converting to probabilities
    if overlapping_classes:
        logits = torch.logit(F.sigmoid(logits).mean(dim=mc_dim), eps=1e-8)
    else:
        # if p = softmax(z) then softmax(log(p)) = p
        # TODO not sure if this is numerically stable
        logits = torch.log(F.softmax(logits, dim=class_dim).mean(dim=mc_dim))
    return logits


def compute_confidence_map(
    logits: torch.Tensor,
    csf_fn: PixelConfidenceScore,
    overlapping_classes: bool,
    mc_dim: int,
    region_aggregation="min",
):
    # logits shape KBCHW[D], if mc_dim=0 (corr. to K)
    class_dim = 1 + 1 * (mc_dim < 2)
    if overlapping_classes:
        assert mc_dim == 0
        assert class_dim == 2  # TODO generalize
        # call csf for every class separately (binary segmentation)
        # and aggregate confidence score across classes
        # (currently only nnunet-style region-based training is supported, which results in a single label per pixel)
        confid = None
        binary_logits_shape = list(logits.shape)
        binary_logits_shape[class_dim] = 2
        for i in range(logits.shape[class_dim]):
            softmax = torch.zeros(binary_logits_shape, device=logits.device)  # allocate KB2HW[D]
            softmax[:, :, 1] = F.sigmoid(logits.select(class_dim, i))
            softmax[:, :, 0] = 1 - softmax[:, :, 1]
            curr_confids = csf_fn(softmax, mc_samples_dim=mc_dim)  # allocate BHW[D]

            if region_aggregation == "min":
                confid = curr_confids if confid is None else torch.minimum(confid, curr_confids)
            elif region_aggregation == "mean":
                confid = curr_confids if i == 0 else (curr_confids + i * confid) / (i + 1)
            else:
                raise ValueError(f"Unknown region aggregation method {region_aggregation}")
    else:
        softmax = F.softmax(logits, dim=class_dim)
        confid = csf_fn(softmax, mc_samples_dim=mc_dim)  # shape BCHW[D]
    return confid
