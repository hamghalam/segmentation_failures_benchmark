from abc import ABC, abstractmethod

import torch

_scores_factories = {}


def register_pixel_csf(name):
    def decorator_(cls):
        _scores_factories[name] = cls
        return cls

    return decorator_


def get_pixel_csf(name, **kwargs):
    return _scores_factories[name](**kwargs)


class PixelConfidenceScore(ABC):
    LABEL_DIM = 1

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, mc_samples_dim: int | None = None
    ) -> torch.Tensor:
        """
        Arguments:
            prediction: expected shape is [num_samples, num_classes, *spatial_dims]
            mc_samples_dim: Dimension where the MC samples are located. For example, based on the expected `prediction` shape, this
                results in a shape of [num_samples, num_mc_samples, num_classes, *spatial_dims] for `mc_samples_dim=1`.
        returns:
            confidences: shape [num_samples, *pixel_shape]
        """


@register_pixel_csf("maxsoftmax")
class MaximumSoftmaxScore(PixelConfidenceScore):
    def __call__(self, softmax: torch.Tensor, mc_samples_dim: int | None = None) -> torch.Tensor:
        label_dim = self.LABEL_DIM
        if mc_samples_dim is not None:
            label_dim = self.LABEL_DIM + (mc_samples_dim <= self.LABEL_DIM)
        if mc_samples_dim is not None:
            # average MC-samples first
            mean_softmax = softmax.mean(dim=mc_samples_dim, keepdim=True)
            return torch.amax(mean_softmax, dim=label_dim).squeeze(dim=mc_samples_dim)
        return torch.amax(softmax, label_dim)


@register_pixel_csf("predictive_entropy")
class PredictiveEntropyScore(PixelConfidenceScore):
    # this computes a shifted and negative entropy to make it a confidence score
    def __call__(self, softmax: torch.Tensor, mc_samples_dim: int | None = None) -> torch.Tensor:
        label_dim = self.LABEL_DIM
        if mc_samples_dim is None:
            entropy = compute_entropy(softmax)
        else:
            label_dim = self.LABEL_DIM + (mc_samples_dim <= self.LABEL_DIM)
            entropy = predictive_entropy(softmax, dim=mc_samples_dim)
        num_classes = softmax.shape[label_dim]
        max_entropy = torch.log(num_classes * torch.ones_like(entropy))
        return max_entropy - entropy  # higher score -> higher confidence


@register_pixel_csf("expected_entropy")
class ExpectedEntropyScore(PixelConfidenceScore):
    # this computes a shifted and negative entropy to make it a confidence score
    def __call__(self, softmax: torch.Tensor, mc_samples_dim: int) -> torch.Tensor:
        # for mc_samples_dim == 0, expected  prediction shape is [n_mc_samples, n_batch, n_classes, *spatial_dims]
        # returns confidence of shape [n_batch, *spatial_dims]
        label_dim = self.LABEL_DIM + (mc_samples_dim <= self.LABEL_DIM)
        if mc_samples_dim is None:
            raise ValueError("Expected entropy is only defined over multiple samples")
        entropy = expected_entropy(softmax, dim=mc_samples_dim)
        num_classes = softmax.shape[label_dim]
        max_entropy = torch.log(num_classes * torch.ones_like(entropy))
        return max_entropy - entropy  # higher score -> higher confidence


@register_pixel_csf("mutual_information")
class MutualInformationScore(PixelConfidenceScore):
    def __call__(self, softmax: torch.Tensor, mc_samples_dim: int) -> torch.Tensor:
        if mc_samples_dim is None:
            raise ValueError("Expected entropy is only defined over multiple samples")
        mutual_info = predictive_entropy(softmax, dim=mc_samples_dim) - expected_entropy(
            softmax, dim=mc_samples_dim
        )
        return -mutual_info


def predictive_entropy(prob_distr: torch.Tensor, dim: int) -> torch.Tensor:
    return compute_entropy(prob_distr.mean(dim=dim))


def expected_entropy(prob_distr: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.concatenate(
        [compute_entropy(p).unsqueeze(dim) for p in list(torch.unbind(prob_distr, dim=dim))]
    ).mean(dim=dim)


def compute_entropy(prob: torch.Tensor):
    # assumed shape: NxD -> entropy along axis 1
    logzero_fix = torch.zeros_like(prob)
    logzero_fix[prob == 0] = torch.finfo(prob.dtype).eps
    return torch.sum(prob * (-torch.log(prob + logzero_fix)), dim=1)
