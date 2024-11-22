import random

import monai
import numpy as np
import radiomics
import SimpleITK as sitk
import torch
from loguru import logger
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from segmentation_failures.models.confidence_aggregation.heuristic import (
    AbstractHeuristicAggregationModule,
    get_regression_model,
)

# from segmentation_failures.utils.visualization import make_image_mask_grid


# deprecated; I switched to sklearn's SimpleImputer for simplicity
# Also, this would need a fit/transform functionality (training/inference time)
def replace_nans(array_: np.ndarray, inplace=True) -> np.ndarray:
    # array shape: (N_samples, N_features)
    if not inplace:
        array_ = array_.copy()
    nan_mask = np.isnan(array_)
    num_nans = nan_mask.sum(axis=0)
    # this inserts random feature values (chosen from existing ones)
    for feature_idx in range(array_.shape[1]):
        if num_nans[feature_idx] == 0:
            continue
        if num_nans[feature_idx] == array_.shape[0]:
            # only nan values for this feature. Set to 0 (or drop?)
            array_[:, feature_idx] = 0
            continue
        valid_idxs = np.arange(len(array_))[~nan_mask[:, feature_idx]]
        rand_idx = random.choices(valid_idxs, k=num_nans[feature_idx])
        curr_col_mask = np.zeros(array_.shape, dtype=bool)
        curr_col_mask[:, feature_idx] = nan_mask[:, feature_idx]
        array_[curr_col_mask] = array_[rand_idx, feature_idx]
    return array_


class RadiomicsAggregationModule(AbstractHeuristicAggregationModule):
    def __init__(
        self,
        regression_model: str,
        dataset_id: int,
        confid_name: str,
        target_metrics: list[str],
        image_dim: int,
        confid_threshold: float | None = None,
        imputation_strategy: str = "mean",
    ) -> None:
        super().__init__(
            regression_model=regression_model,
            dataset_id=dataset_id,
            confid_name=confid_name,
            target_metrics=target_metrics,
        )
        self.save_hyperparameters()
        self.radiomics_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.quality_estimator = make_pipeline(
            # Keeping it simple for now.
            SimpleImputer(
                missing_values=np.nan,
                keep_empty_features=True,
                strategy=imputation_strategy,
            ),
            StandardScaler(),
            get_regression_model(regression_model),
        )
        self.auto_threshold = False
        if confid_threshold is None:
            self.auto_threshold = True
            confid_threshold = 0.5  # This is just a dummy value to allow lightning's sanity checks
        self.register_buffer("confid_thresh", torch.tensor(confid_threshold))

        self.feature_list = []
        for class_name, feature_cls in radiomics.getFeatureClasses().items():
            full_feature_names = [
                f"original_{class_name}_{x}" for x in feature_cls.getFeatureNames()
            ]
            self.feature_list.extend(full_feature_names)
        if image_dim == 2:
            self.radiomics_extractor.enableFeatureClassByName("shape2D")
            self.radiomics_extractor.enableFeatureClassByName("shape", False)

    def extract_features(self, prediction, pxl_confid):
        # prediction and confid shape BHW[D]
        # compute ROI based on pxl_confid
        pxl_confid = pxl_confid.cpu()
        extractor_roi = (pxl_confid < self.confid_thresh) * 1
        # if self.trainer.validating:
        #     self.log_confid_rois(image, prediction, extractor_roi)
        all_features = torch.ones(len(pxl_confid), len(self.feature_list)) * torch.nan
        # missing values will be none
        for sample_idx in range(len(pxl_confid)):
            # apparently, pyradiomics needs an sitk image, so we'll convert here:
            curr_roi = extractor_roi[sample_idx]
            curr_confid = pxl_confid[sample_idx]
            if curr_roi.sum() < 5:
                logger.warning(
                    "When extracting radiomics features, encountered samples with an ROI < 5 => inserting NaNs."
                )
                continue
            if curr_roi.float().mean() == 1:
                logger.warning(
                    "When extracting radiomics features, encountered samples with an ROI of all 1s => inserting NaNs."
                )
                continue
            sitk_roi = sitk.GetImageFromArray(curr_roi.numpy())
            sitk_pxl_confid = sitk.GetImageFromArray(curr_confid.numpy())
            feature_dict = self.radiomics_extractor.execute(sitk_pxl_confid, sitk_roi)
            # There are also "diagnostic" entries in the dict, which do not contain actual radiomics features
            for feat_idx, feat_name in enumerate(self.feature_list):
                if feat_name in feature_dict:
                    all_features[sample_idx, feat_idx] = torch.tensor(feature_dict[feat_name])
        return all_features

    def on_train_start(self) -> None:
        if self.auto_threshold:
            # otherwise I respect the value set by the user.
            best_thresh = self.tune_confid_threshold(self.trainer.val_dataloaders)
            self.confid_thresh = torch.tensor(best_thresh)
            logger.info(f"Auto-tuned confid_thresh to {best_thresh}.")

    @torch.no_grad()
    def tune_confid_threshold(self, val_dataloader):
        # Find confidence range first
        min_confid = np.inf
        max_confid = -np.inf
        for batch in val_dataloader:
            min_confid = min(min_confid, batch["confid"].min())
            max_confid = max(max_confid, batch["confid"].max())
        # Jungo et al normalize the confidence values to [0, 1] and scan thresholds in [0.05, 0.95]
        try_thresholds = torch.linspace(0.05, 0.95, 100) * (max_confid - min_confid) + min_confid
        sum_ue_vals = torch.zeros_like(try_thresholds)
        for batch in val_dataloader:
            tmp_ue_list = torch.zeros(len(try_thresholds), len(batch["pred"]))
            error_mask = batch["pred"] != batch["target"]  # shape B1HW[D]
            for thresh_idx, curr_thresh in enumerate(try_thresholds):
                uncertain_mask = (batch["confid"] < curr_thresh).to(error_mask)
                # convert masks to one-hot encoding and permute so that class dim is second
                ue_dice = monai.metrics.compute_dice(
                    uncertain_mask, error_mask, include_background=True, ignore_empty=False
                )
                tmp_ue_list[thresh_idx] = ue_dice.cpu()
            sum_ue_vals += tmp_ue_list.sum(dim=1)
        return try_thresholds[torch.argmax(sum_ue_vals)]

    # def log_confid_rois(self, images, predictions, rois):
    #     # TODO does this work for 3D?
    #     if isinstance(images, monai.data.MetaTensor):
    #         images = images.as_tensor()
    #     if isinstance(predictions, monai.data.MetaTensor):
    #         predictions = predictions.as_tensor()
    #     if isinstance(rois, monai.data.MetaTensor):
    #         rois = rois.as_tensor()
    #     # convert predictions and rois to one-hot encoding
    #     predictions = F.one_hot(
    #         torch.argmax(predictions, dim=1), num_classes=self.hparams.num_classes
    #     )
    #     rois = F.one_hot(rois.long(), num_classes=2)
    #     # permute so that class dim is second
    #     predictions = torch.permute(predictions, (0, -1, *range(1, predictions.ndim - 1)))
    #     rois = torch.permute(rois, (0, -1, *range(1, rois.ndim - 1)))
    #     rgb_grid = make_image_mask_grid(images, [predictions, rois], max_images=5, alpha=0.5)
    #     for expt_logger in self.loggers:
    #         if hasattr(expt_logger.experiment, "add_image"):
    #             expt_logger.experiment.add_image("confid_rois", rgb_grid, self.current_epoch)
