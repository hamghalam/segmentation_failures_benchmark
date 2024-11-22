from enum import Enum

import torchio as tio


class TransformMagnitude(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TransformRegistry:
    transforms = {}

    @classmethod
    def register(cls, name: str):
        if name in cls.transforms:
            print("WARNING: Transform %s already registered. Will replace it", name)

        def inner(func):
            cls.transforms[name] = func
            return func

        return inner

    @classmethod
    def get_transform(cls, name: str, magnitude: str, **kwargs):
        magnitude = TransformMagnitude[magnitude.upper()]
        # kwargs can overwrite the magnitude settings
        if name not in cls.transforms:
            raise ValueError("Transform %s not registered" % name)
        transform, trf_settings = cls.transforms[name](magnitude)
        trf_settings.update(kwargs)
        return transform(**trf_settings), trf_settings

    @classmethod
    def list_transforms(cls):
        return list(cls.transforms)


@TransformRegistry.register("biasfield")
def biasfield(magnitude: TransformMagnitude):
    default_settings = {
        "order": 3,
    }
    magnitude_overwrites = {
        TransformMagnitude.LOW: {"coefficients": 0.2},
        TransformMagnitude.MEDIUM: {"coefficients": 0.6},
        TransformMagnitude.HIGH: {"order": 5, "coefficients": 0.8},
    }
    default_settings.update(magnitude_overwrites[magnitude])
    return tio.RandomBiasField, default_settings


@TransformRegistry.register("ghosting")
def ghosting(magnitude: TransformMagnitude):
    default_settings = {
        "num_ghosts": (4, 8),
    }
    magnitude_overwrites = {
        TransformMagnitude.LOW: {"intensity": (0.2, 0.4)},
        TransformMagnitude.MEDIUM: {"intensity": (0.5, 0.7)},
        TransformMagnitude.HIGH: {"intensity": (0.8, 1.0)},
    }
    default_settings.update(magnitude_overwrites[magnitude])
    return tio.RandomGhosting, default_settings


@TransformRegistry.register("spike")
def spike(magnitude: TransformMagnitude):
    default_settings = {
        "num_spikes": 1,
    }
    magnitude_overwrites = {
        TransformMagnitude.LOW: {"intensity": (0.1, 0.2)},
        TransformMagnitude.MEDIUM: {"intensity": (0.3, 0.5)},
        TransformMagnitude.HIGH: {"intensity": (0.7, 0.9)},
    }
    # negative intensities aren't used here because tio would then sample from [-I, I] and
    # I want the artefacts to be always visible
    default_settings.update(magnitude_overwrites[magnitude])
    return tio.RandomSpike, default_settings


@TransformRegistry.register("affine")
def affine(magnitude: TransformMagnitude):
    default_settings = {
        "translation": 0,
    }
    magnitude_overwrites = {
        TransformMagnitude.LOW: {"degrees": 5, "scales": (0.9, 1.4)},
        TransformMagnitude.MEDIUM: {"degrees": (5, 15), "scales": (0.7, 1.8)},
        TransformMagnitude.HIGH: {"degrees": (15, 30), "scales": (0.6, 2.0)},
    }
    default_settings.update(magnitude_overwrites[magnitude])
    return tio.RandomAffine, default_settings


# TODO for 2D this transform is not working as expected.
# The results look just like an affine transform
# @TransformRegistry.register("motion")
# def motion(magnitude: TransformMagnitude):
#     default_settings = {
#     }
#     magnitude_overwrites = {
#         TransformMagnitude.LOW: {},
#         TransformMagnitude.MEDIUM: {},
#         TransformMagnitude.HIGH: {},
#     }
#     default_settings.update(magnitude_overwrites[magnitude])
#     return tio.RandomMotion, default_settings


# TODO This transform depends on the image size and spacing, so it's hard to set general values here
# @TransformRegistry.register("elastic")
# def elastic(magnitude: TransformMagnitude):
#     default_settings = {
#         "num_control_points": 7,
#     }
#     magnitude_overwrites = {
#         TransformMagnitude.LOW: {"max_displacement": 2},
#         TransformMagnitude.MEDIUM: {"max_displacement": 3.5},
#         TransformMagnitude.HIGH: {"max_displacement": 4.5},
#     }
#     default_settings.update(magnitude_overwrites[magnitude])
#     return tio.RandomElasticDeformation, default_settings
