from abc import ABC, abstractmethod

import numpy as np
import torch
from monai.metrics import compute_dice, compute_generalized_dice
from scipy import ndimage
from scipy.signal import convolve
from skimage.measure import label
from skimage.segmentation import expand_labels, find_boundaries

_SIMPLE_AGG_REGISTRY = dict()
_AGGREGATOR_REGISTRY = dict()


def register_agg_fn(name):
    def decorator_register(agg_fn):
        _SIMPLE_AGG_REGISTRY[name] = agg_fn
        return agg_fn

    return decorator_register


def register_aggregator(name):
    def decorator_register(aggregator):
        _AGGREGATOR_REGISTRY[name] = aggregator
        return aggregator

    return decorator_register


def get_simple_agg_fn(name):
    return _SIMPLE_AGG_REGISTRY[name]


def get_aggregator(name, **kwargs):
    return _AGGREGATOR_REGISTRY[name](**kwargs)


# Simple functions
@register_agg_fn("mean")
def simple_mean(confidence):
    return torch.mean(confidence.flatten(start_dim=1), dim=1)


@register_agg_fn("sum")
def simple_sum(confidence):
    return torch.sum(confidence.flatten(start_dim=1), dim=1)


@register_agg_fn("logsum")
def simple_logsum(confidence):
    return torch.sum(torch.log(confidence.flatten(start_dim=1)), dim=1)


@register_agg_fn("std")
def simple_std(confidence):
    return torch.std(confidence.flatten(start_dim=1), dim=1)


@register_agg_fn("max")
def simple_max(confidence):
    return torch.amax(confidence.flatten(start_dim=1), dim=1)


@register_agg_fn("min")
def simple_min(confidence):
    return torch.amin(confidence.flatten(start_dim=1), dim=1)


@register_agg_fn("p95")
def simple_quantile95(confidence):
    return torch.quantile(confidence.flatten(start_dim=1), dim=1, q=0.95)


@register_agg_fn("p05")
def simple_quantile05(confidence):
    return torch.quantile(confidence.flatten(start_dim=1), dim=1, q=0.05)


class AbstractAggregator(ABC):
    def __call__(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """Abstract interface for confidence aggregators.

        Args:
            prediction (torch.Tensor): Assumes prediction logits of shape [batch, *spatial_dims].
            confidence (torch.Tensor): Assumes pixel confidence of shape [batch, *spatial_dims]

        Returns:
            torch.Tensor: scalar confidence score for each batch item, shape [batch,]
        """
        self.validate_input(prediction, confidence)
        return self.aggregate(prediction, confidence)

    @abstractmethod
    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def validate_input(prediction: torch.Tensor, confidence: torch.Tensor) -> None:
        if prediction.shape != confidence.shape:
            raise ValueError(
                f"Prediction and confidence shape mismatch: {prediction.shape} vs {confidence.shape}"
            )
        if len(prediction.shape) > 4:
            raise ValueError("Prediction and confidence should have at most 4 dimensions (BHWD).")
        if prediction.dtype.is_floating_point:
            raise ValueError("Prediction should be labels, not probabilities.")


class AbstractEnsembleAggregator(ABC):
    def __call__(self, label_distr: torch.Tensor) -> torch.Tensor:
        """Abstract interface for confidence aggregators.

        Args:
            label_distr (torch.Tensor): Assumes prediction logits of shape [samples, batch, class, *spatial_dims].
                Assume boolean tensor.

        Returns:
            torch.Tensor: scalar confidence score for each batch item, shape [batch,]
        """
        self.validate_input(label_distr)
        return self.aggregate(label_distr)

    @abstractmethod
    def aggregate(self, label_distr: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def validate_input(label_distr: torch.Tensor) -> None:
        if label_distr.ndim > 6:
            raise ValueError("Input should have at most 6 dimensions (NBCHWD).")
        if label_distr.dtype != torch.bool:
            raise ValueError("Prediction should be boolean labels.")


@register_aggregator("pairwise_dice")
class PairwiseDiceAggregator(AbstractEnsembleAggregator):
    def __init__(
        self, include_zero_label: bool, gen_dice_weight: str = "square", use_mean_dice=False
    ):
        # square weighting is default in monai
        self.include_background = include_zero_label
        self.gdice_weight = gen_dice_weight
        self.use_mean_dice = use_mean_dice
        if gen_dice_weight not in ["simple", "square", "uniform"]:
            raise ValueError()

    def aggregate(self, label_distr: torch.Tensor) -> torch.Tensor:
        # input label_distr shape NBCHW[D] -> BNCHW[D]
        label_distr = label_distr.permute(1, 0, *range(2, label_distr.ndim))
        batch_size = label_distr.shape[0]  # often this is 1 (inference)
        num_mc_samples = label_distr.shape[1]
        if num_mc_samples <= 1:
            raise ValueError("Pairwise dice is only defined over multiple samples")
        pw_dice_scores = torch.zeros(batch_size, device=label_distr.device)  # shape B
        for batch_idx in range(batch_size):
            counter = 0
            for mc_idx1 in range(num_mc_samples):
                for mc_idx2 in range(num_mc_samples):
                    if mc_idx2 >= mc_idx1:
                        continue
                    # 0th class is background for exclusive labels, fg class else
                    if self.use_mean_dice:
                        dice_score = compute_dice(
                            label_distr[batch_idx, mc_idx1].unsqueeze(dim=0),
                            label_distr[batch_idx, mc_idx2].unsqueeze(dim=0),
                            include_background=self.include_background,
                            ignore_empty=False,  # otherwise I get NaNs
                        )
                        dice_score = dice_score.mean(dim=1)
                    else:
                        dice_score = compute_generalized_dice(
                            label_distr[batch_idx, mc_idx1].unsqueeze(dim=0),
                            label_distr[batch_idx, mc_idx2].unsqueeze(dim=0),
                            include_background=self.include_background,
                            weight_type=self.gdice_weight,
                        )
                    pw_dice_scores[batch_idx] += dice_score.squeeze()
                    counter += len(dice_score)
            pw_dice_scores[batch_idx] /= counter  # or * 2 / (n * (n-1))
            counter = 0
        return pw_dice_scores


@register_aggregator("simple")
class SimpleAggregator(AbstractAggregator):
    def __init__(self, agg_fn: str = None) -> None:
        """Aggregates confidence by applying a simple function.

        Args:
            agg_fn (Callable, optional): Function to aggregate the masked confidence with. Defaults to mean.
        """
        if agg_fn is None:
            agg_fn = "mean"
        self.agg_fn = get_simple_agg_fn(agg_fn)

    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        return self.agg_fn(confidence)


class WeightedAggregator(AbstractAggregator):
    def __init__(self, agg_fn: str = None, normalize_weights=True) -> None:
        """Aggregates confidence by weighting each voxel and applying an aggregation function.

        The specific weighting and aggregation function can be adjusted through inheritance/configuration.
        Args:
            agg_fn (Callable, optional): Function to aggregate the masked confidence with. Defaults to mean.
        """
        if agg_fn is None:
            agg_fn = "sum"  # I use sum here so that the default is a weighted average
        self.agg_fn = get_simple_agg_fn(agg_fn)
        self.normalize_weights = normalize_weights

    @abstractmethod
    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        """Compute mask based on prediction outputs.

        Args:
            single_prediction (torch.Tensor): Prediction of shape (C, *spatial_dims), where C is the one-hot class channel.

        Returns:
            torch.Tensor: mask to use for pixel-confidence
        """

    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(confidence, device=confidence.device)
        for i, y in enumerate(prediction):
            # TODO can this be vectorized?
            curr_weights = self.compute_weights(y)
            assert curr_weights.shape == confidence.shape[1:]
            if curr_weights.sum() == 0:
                # avoid zero division
                curr_weights = torch.ones_like(curr_weights)
            weights[i] = curr_weights
        if self.normalize_weights:
            weights = weights / torch.sum(weights, dim=list(range(1, weights.ndim)), keepdim=True)
        weighted_confidence = weights * confidence
        # NOTE If necessary I could extend the agg_fn so that it takes mask & confidence and does some spatial stuff.
        return self.agg_fn(weighted_confidence)


@register_aggregator("boundary_weighted")
class BoundaryAggregator(WeightedAggregator):
    def __init__(self, agg_fn=None, boundary_width=2, invert=False) -> None:
        """Aggregates confidence only on the boundary/non-boundary regions.

        Args:
            boundary_width (int, optional): Include pixels closer than this to the boundary. Defaults to 2 (2 pxl wide boundary).
            invert (bool, optional): Invert the boundary mask. Defaults to False.
        """
        super().__init__(agg_fn)
        if boundary_width < 2:
            raise ValueError("Boundary width must be at least 2.")
        if boundary_width % 2 == 1:
            raise ValueError("Boundary width must be even.")
        # by default the boundary is 2 pxl wide. To expand it to width W, we need to add self.expand_bd pixels at each side.
        self.expand_bd = int((boundary_width - 2) / 2)
        self.invert = invert

    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        # input dim HW[D]
        boundary = find_boundaries(single_prediction.cpu().numpy()) * 1.0
        if self.expand_bd > 0:
            boundary = expand_labels(boundary, distance=self.expand_bd)
        if self.invert:
            boundary = 1 - boundary
        return torch.from_numpy(boundary).to(single_prediction.device)


@register_aggregator("distance_weighted")
class EuclideanDistanceMapAggregator(WeightedAggregator):
    """This follows the `distance weighting` aggregation from Jungo et al. (2020)"""

    def __init__(
        self, agg_fn=None, spacing: float | list[float] | None = None, saturate: float | None = 4.0
    ) -> None:
        """Weights voxel confidences proportional to the distance to the boundary.

        Args:
            spacing (float | list[float], optional): Voxel spacing in each dimension. Defaults to None.
            saturate (float, optional): Distance at which weight saturates. Defaults to 4.0 (times the spacing if given).
        """
        super().__init__(agg_fn)
        self.spacing = spacing
        self.saturate = saturate

    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        # input dim [N_classes, *spatial]
        boundary = find_boundaries(single_prediction.cpu().numpy())
        edm = ndimage.distance_transform_edt(1 - boundary, sampling=self.spacing)
        if self.saturate is not None:
            max_val = self.spacing if self.spacing is not None else 1
            max_val *= self.saturate
            edm = np.clip(edm, a_min=0, a_max=max_val)
        return torch.from_numpy(edm).to(single_prediction.device)


@register_aggregator("foreground_weighted")
class ForegroundAggregator(WeightedAggregator):
    def __init__(self, agg_fn=None, boundary_width=0) -> None:
        """Aggregates confidence only in the foreground pixels.

        Args:
            boundary_width (int, optional): Include pixels closer than this to the boundary. Defaults to 0.
        """
        super().__init__(agg_fn)
        self.boundary_agg = None
        if boundary_width > 0:
            self.boundary_agg = BoundaryAggregator(boundary_width=boundary_width)

    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        # input dim [N_classes, *spatial]
        interior_mask = (single_prediction > 0).to(torch.float32)
        if self.boundary_agg:
            boundary = self.boundary_agg.compute_weights(single_prediction)
            interior_mask = interior_mask * (1 - boundary)
        return interior_mask


@register_aggregator("background_weighted")
class BackgroundAggregator(WeightedAggregator):
    def __init__(self, agg_fn=None, boundary_width=2) -> None:
        """Complements ForegroundAggregator.

        Args:
            boundary_width (int, optional): Include pixels closer than this to the boundary. Defaults to 0.
        """
        super().__init__(agg_fn)
        self.boundary_agg = None
        if boundary_width > 0:
            self.boundary_agg = BoundaryAggregator(boundary_width=boundary_width)

    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        # input dim [N_classes, *spatial]
        bg_interior_mask = (single_prediction == 0).to(torch.float32)
        if self.boundary_agg:
            boundary = self.boundary_agg.compute_weights(single_prediction)
            bg_interior_mask = bg_interior_mask * (1 - boundary)
        return bg_interior_mask


# This is not really confidence aggregation, rather a simple feature extraction
# but the heuristic module currently needs this interface
@register_aggregator("fg_size")
class ForegroundSizeAggregator(AbstractAggregator):
    def __init__(self, fractional_size=False) -> None:
        self.fractional_size = fractional_size

    # Neglects confidence and just computes the total size/fraction of foreground
    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        foreground_mask = prediction != 0
        if self.fractional_size:
            return foreground_mask.flatten(start_dim=1).float().mean(dim=1)
        return foreground_mask.flatten(start_dim=1).float().sum(dim=1)


@register_aggregator("fg_volume_weighted")
class ForegroundSizeWeightedAggregator(WeightedAggregator):
    """This follows the `volume weighting` aggregation from Jungo et al. (2020)"""

    def __init__(self) -> None:
        # This is fixed for this aggregator
        super().__init__(agg_fn="mean", normalize_weights=False)

    def compute_weights(self, single_prediction: torch.Tensor) -> torch.Tensor:
        foreground_mask = single_prediction != 0
        foreground_size = foreground_mask.sum()  # in voxels
        if foreground_size == 0:
            # avoid zero division
            foreground_size = 1
        return torch.ones_like(foreground_mask) / foreground_size


# This is not really confidence aggregation, rather a simple feature extraction
# but the heuristic module currently needs this interface
@register_aggregator("connected_components")
class ConnectedComponentsAggregator(AbstractAggregator):
    # Neglects confidence and just computes the number of CC
    # Other Possible analyses based on CC:
    # - largest distance between CCs
    # - median (?) size of CCs
    # - something which uses confidence?
    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        output_confid = []
        for y in prediction:
            _, num_cc = label(y.cpu().numpy(), background=0, return_num=True)
            output_confid.append(num_cc)
        return torch.tensor(output_confid, device=confidence.device)


@register_aggregator("patch_min")
class PatchMinAggregator(AbstractAggregator):
    def __init__(self, patch_size: tuple[int] | int = None, mean=True):
        self.patch_size = patch_size
        self.mean = mean

    def aggregate(self, prediction: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        img_dim = len(confidence.shape) - 1
        if self.patch_size is None:
            self.patch_size = 10
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size] * img_dim
        confid_np = confidence.detach().cpu().numpy()
        agg_confid = np.zeros(confid_np.shape[0])
        # TODO inefficient; vectorize
        for batch_idx in range(confid_np.shape[0]):
            agg_confid[batch_idx] = patch_level_aggregation(
                confid_np[batch_idx], self.patch_size, self.mean
            )
        return torch.tensor(agg_confid).to(confidence.device)


# Adapted from:
# https://github.com/IML-DKFZ/values/blob/main/evaluation/uncertainty_aggregation/aggregate_uncertainties.py
def patch_level_aggregation(image, patch_size, mean=True):
    # if img_size is smaller than patch_size in one dimension, reduce patch_size
    patch_size = np.minimum(patch_size, image.shape)
    kernel = np.ones(patch_size)
    patch_aggregated = convolve(image, kernel, mode="valid")
    if mean:
        patch_aggregated = patch_aggregated / (np.prod(patch_size))
    return float(np.min(patch_aggregated))
    # all_max_indices = np.where(np.isclose(patch_aggregated, np.max(patch_aggregated)))
    # max_indices = []
    # for indices in all_max_indices:
    #     max_indices.append(indices[0])

    # max_indices_slice = []
    # for idx, index in enumerate(max_indices):
    #     max_indices_slice.append((int(index), int(index + patch_size[idx])))
    # return float(np.max(patch_aggregated)), max_indices_slice
