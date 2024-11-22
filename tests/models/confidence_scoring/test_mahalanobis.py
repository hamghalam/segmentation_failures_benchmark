import numpy as np
import torch

from segmentation_failures.models.image_confidence import SingleGaussianOODDetector


def test_fit_gaussian():
    toy_features = torch.randn(100, 2) + 42
    dummy_module = torch.nn.Module()
    dummy_module.add_module(
        "model", torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    )
    # Prepare mock objects
    my_ood_detector = SingleGaussianOODDetector(
        segmentation_net=dummy_module,
        feature_path="0",
    )
    my_ood_detector.training_epoch_end(outputs=[{"features": toy_features}])

    assert np.all(
        my_ood_detector.gaussian_estimator.location_ == toy_features.numpy().mean(axis=0)
    )


# def test_save_load_model():
#     feature_path = "dummy"
#     toy_data = torch.randn(100, 2) + 42
#     expected_location = toy_data.numpy().mean(axis=0)

#     # Prepare mock objects
#     my_ood_detector = SingleGaussianOODDetector(feature_path=feature_path)
#     my_ood_detector.training_epoch_end(outputs=[{"features": toy_data}])

#     assert np.all(my_ood_detector.gaussian_estimator.location_ == expected_location)

#     # # TODO idk how to save/load manually
#     # SingleGaussianOODDetector.save -> need lightning trainer for this...
#     # SingleGaussianOODDetector.load_from_checkpoint()
