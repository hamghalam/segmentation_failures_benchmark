from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from segmentation_failures.utils import GLOBAL_SEEDS

EXPERIMENT_TASKS = (
    "train_seg",
    "validate_pixel_csf",
    "test_pixel_csf",
    "train_image_csf",
    "test_fd",
)


@dataclass
class Experiment:
    # Can add more if required
    group: str
    task: str
    dataset: str
    fold: int
    backbone: str
    segmentation: str
    seed: int = 0
    csf_pixel: str | None = None  # this is softmax, mcdropout, ensemble, ...
    csf_image: str | None = None
    csf_aggregation: str | None = None
    config_path: str = None
    # config_path is usually not set. It is only set if the experiment is created manually.

    def __post_init__(self):
        if self.task not in EXPERIMENT_TASKS:
            raise ValueError(f"Task {self.task} not supported")
        if (
            self.task in ["train_image_csf", "test_fd"]
            and self.csf_image is None
            and self.csf_aggregation is None
        ):
            raise ValueError("Need to specify a csf method for confidence training/testing")

    @property
    def name(self):
        expt_name = f"{self.segmentation}-{self.backbone}"
        if self.is_seg_experiment():
            # I don't include the csf stuff in the name for segmentation experiments
            return expt_name
        expt_name += f"-{self.csf_pixel}"  # even if this is None
        if self.csf_aggregation is not None:
            expt_name += f"-{self.csf_aggregation}"
        if self.csf_image is not None:
            expt_name += f"-{self.csf_image}"
        return expt_name

    def get_seed(self):
        if self.seed not in GLOBAL_SEEDS:
            raise ValueError(f"Seed not in the set of predefined seeds: {list(GLOBAL_SEEDS)}")
        return GLOBAL_SEEDS[self.seed]

    @staticmethod
    def from_config_path(config_path: str):
        return Experiment(
            group=None,
            dataset=None,
            fold=None,
            backbone=None,
            segmentation=None,
            csf_pixel=None,
            csf_image=None,
            csf_aggregation=None,
            config_path=config_path,
        )

    def is_seg_experiment(self):
        return self.task == "train_seg"

    def overwrites(self):
        overwrite_dict = {
            "segmentation": self.segmentation,
            "backbone": self.backbone,
            "dataset": self.dataset,
            "datamodule": get_datamodule_config(self),
            "datamodule.fold": self.fold,
            "seed": self.get_seed(),
        }
        overwrite_dict["expt_group"] = self.group
        if self.csf_image is not None:
            overwrite_dict["csf_image"] = self.csf_image
            # expect format {measure}+{aggregation}
            # measure has to match one of the confidence names returned by the pixel CSF model
            csf_str = self.csf_image.split("+")
            if len(csf_str) == 2:
                # if aggregation works only with a single csf
                overwrite_dict["csf_image"] = csf_str[1]
                overwrite_dict["csf_image.hparams.confid_name"] = csf_str[0]
            else:
                overwrite_dict["csf_image"] = csf_str[0]
        if self.csf_pixel is not None:
            overwrite_dict["csf_pixel"] = self.csf_pixel
        if self.csf_aggregation is not None:
            # expect format {measure}+{aggregation}
            # measure has to match one of the confidence names returned by the pixel CSF model
            agg_str = self.csf_aggregation.split("+")
            if len(agg_str) == 2:
                # if aggregation works only with a single csf
                overwrite_dict["csf_aggregation"] = agg_str[1]
                overwrite_dict["csf_aggregation.hparams.confid_name"] = agg_str[0]
            else:
                overwrite_dict["csf_aggregation"] = agg_str[0]

        # needs to happen after the above block, iirc
        overwrite_dict["expt_name"] = self.name
        # can add more custom overwrites if necessary
        # especially for special cases (combinations of dataset and model components)
        overwrite_dict.update(self._special_combination_overwrites(overwrite_dict))
        return overwrite_dict

    def _special_combination_overwrites(self, initial_overwrites: dict[str, Any]):
        # NOTE this could also be done using hydra's specialization, https://hydra.cc/docs/patterns/specializing_config/
        # but this is easier to understand
        overwrites = {}
        # training length for different architectures/datasets
        if "monai_unet" in self.backbone and "simple_fets" in self.dataset:
            overwrites["backbone.hparams.norm"] = "batch"
            if self.is_seg_experiment():
                overwrites["trainer.max_epochs"] = 1000
                overwrites["trainer.check_val_every_n_epoch"] = 10
        if self.csf_pixel is not None and self.dataset in [
            "simple_fets22_corrupted",
            "mnms",
        ]:
            overwrites["csf_pixel.hparams.everything_on_gpu"] = True
        if self.is_seg_experiment() and "acdc" in self.dataset:
            overwrites["trainer.max_epochs"] = 1000
            overwrites["trainer.check_val_every_n_epoch"] = 3
        if self.is_seg_experiment() and "covid" in self.dataset:
            overwrites["trainer.max_epochs"] = 3000
            overwrites["trainer.check_val_every_n_epoch"] = 10
        if self.is_seg_experiment() and "mnms" in self.dataset:
            overwrites["trainer.max_epochs"] = 2500
            overwrites["trainer.check_val_every_n_epoch"] = 5
        if self.is_seg_experiment() and "prostate" in self.dataset:
            overwrites["trainer.max_epochs"] = 7500
            overwrites["trainer.check_val_every_n_epoch"] = 15
        if self.is_seg_experiment() and "kits23" in self.dataset:
            overwrites["trainer.max_epochs"] = 2000
            overwrites["trainer.check_val_every_n_epoch"] = 5
        if self.is_seg_experiment() and "brats19_lhgg" in self.dataset:
            overwrites["trainer.max_epochs"] = 1500
            overwrites["trainer.check_val_every_n_epoch"] = 5
        if self.is_seg_experiment() and "retouch" in self.dataset:
            overwrites["trainer.max_epochs"] = 10000
            overwrites["trainer.check_val_every_n_epoch"] = 20
        if self.is_seg_experiment() and "mvseg" in self.dataset:
            overwrites["trainer.max_epochs"] = 6000
            overwrites["trainer.check_val_every_n_epoch"] = 8
        if self.is_seg_experiment() and "octa500" in self.dataset:
            overwrites["trainer.max_epochs"] = 2000
            overwrites["trainer.check_val_every_n_epoch"] = 2
        if self.is_seg_experiment() and "retina" in self.dataset:
            overwrites["trainer.max_epochs"] = 500
            overwrites["trainer.check_val_every_n_epoch"] = 5
        if self.csf_aggregation == "all_simple":
            overwrites["trainer"] = "cpu"
        # Mahalanobis stuff
        if self.csf_image is not None and self.csf_image.startswith("mahalanobis"):
            if self.backbone in ["monai_unet", "monai_unet_dropout"]:
                overwrites["csf_image.hparams.feature_path"] = (
                    "model.1.submodule.1.submodule.1.submodule.1.submodule.residual"
                )
            elif self.backbone == "dynamic_unet":
                overwrites["csf_image.hparams.feature_path"] = "bottleneck.blocks.0.conv2"
            elif self.backbone in ["dynamic_unet_dropout", "dynamic_wideunet_dropout"]:
                overwrites["csf_image.hparams.feature_path"] = "bottleneck.0.blocks.0.conv2"
            elif self.backbone == "dynamic_resencunet_dropout":
                overwrites["csf_image.hparams.feature_path"] = "bottleneck.0.blocks.2.conv2"
            if self.csf_image.endswith("gonzalez"):
                # Need to use a special dataloader configuration in this case
                if self.dataset != "simple_fets22_corrupted":
                    overwrites["datamodule.preproc_only"] = True
                    overwrites["datamodule.batch_size"] = 1
        # Learned confidence aggregation
        if self.csf_aggregation is not None and (
            "radiomics" in self.csf_aggregation or "heuristic" in self.csf_aggregation
        ):
            overwrites["datamodule.confid_name"] = initial_overwrites[
                "csf_aggregation.hparams.confid_name"
            ]
            if self.task == "train_image_csf":
                pxl_expt_name = "-".join(self.name.split("-")[:-1])
                folds_str = "-".join(
                    [str(self.fold // 5 + i) for i in range(5)]
                )  # assume 5-fold CV
                train_pred_dir = (
                    f"heuristic_radiomics/{pxl_expt_name}/folds{folds_str}_seed_{self.seed}"
                )
                overwrites["datamodule.prediction_dir"] = train_pred_dir
                overwrites["datamodule.confid_dir"] = train_pred_dir
        if self.csf_image is not None and self.csf_image.endswith("quality_regression"):
            # This solution is not so nice, but this way I can specify the
            # prediction_dir via csf_pixel (necessary for training)
            # pixel csf is needed (even if none) for a reasonable standardization
            tmp_expt = deepcopy(self)
            tmp_expt.csf_image = None
            if self.dataset == "retina":
                overwrites["+datamodule.nnunet_configuration"] = "2d"
            if self.task == "train_image_csf":
                folds_str = "-".join(
                    [str(self.fold // 5 + i) for i in range(5)]
                )  # assume 5-fold CV
                overwrites["datamodule.prediction_dir"] = (
                    tmp_expt.name + f"/folds{folds_str}_seed_{self.seed}"
                )
                if self.dataset == "simple_fets22_corrupted":
                    overwrites["datamodule.batch_size"] = 32
                elif self.dataset == "mnms":
                    overwrites["datamodule.batch_size"] = 4
                elif self.dataset == "mvseg23":
                    overwrites["datamodule.batch_size"] = 6
                elif self.dataset == "retina":
                    overwrites["datamodule.batch_size"] = 32
            if "+" in self.csf_image:
                overwrites["datamodule.confid_name"] = self.csf_image.split("+")[0]
                if self.task == "train_image_csf":
                    overwrites["datamodule.confid_dir"] = tmp_expt.name
            if "kits" in self.dataset and "test" in self.task:
                overwrites["datamodule.cache_num"] = 0
        # VAE stuff
        elif self.csf_image is not None and self.csf_image.lower().startswith("vae"):
            if self.dataset == "simple_fets22_corrupted":
                overwrites["datamodule.batch_size"] = 32
            elif self.dataset == "mvseg23":
                overwrites["datamodule.batch_size"] = 16
            if self.dataset == "retina":
                overwrites["+datamodule.nnunet_configuration"] = "2d"
                overwrites["datamodule.batch_size"] = 32
            if "kits" in self.dataset:
                # the training crashes on my workstation when using batch size > 1 and num_workers > 2
                overwrites["datamodule.num_workers"] = 2
                overwrites["csf_image.hparams.model_h_size"] = [16, 32, 64, 128, 256, 512]
                overwrites["trainer.max_epochs"] = 500  # takes a long time
                if "test" in self.task:
                    overwrites["datamodule.cache_num"] = 0
            if "covid" in self.dataset:
                overwrites["csf_image.hparams.model_h_size"] = [16, 32, 64, 128, 256, 512]
        # nnUNet stuff
        if self.dataset == "kits23":
            if self.csf_pixel == "mcdropout":
                overwrites["csf_pixel.hparams.num_mcd_samples"] = (
                    5  # more isn't feasible memory-wise
                )
        # save pixel stuff
        if self.task == "validate_pixel_csf":
            overwrites["+callbacks/validate"] = "confidence_saver"
        elif self.task == "test_pixel_csf":
            overwrites["+callbacks/test"] = "confidence_saver"
            if self.csf_pixel in ["mcdropout", "deep_ensemble"]:
                overwrites["+callbacks/test"] = "[confidence_saver,ensemble_prediction_saver]"
        return overwrites

    @staticmethod
    def from_iterables(
        group: str | None,
        task: str,
        dataset: Iterable[str],
        fold: Iterable[int],
        seed: Iterable[int],
        backbone: Iterable[str],
        segmentation: Iterable[str],
        csf_pixel: Iterable[str | None],
        csf_image: Iterable[str | None],
        csf_aggregation: Iterable[str | None],
    ):
        return list(
            map(
                lambda args: Experiment(*args),
                product(
                    (group,),
                    (task,),
                    dataset,
                    fold,
                    backbone,
                    segmentation,
                    seed,
                    csf_pixel,
                    csf_image,
                    csf_aggregation,
                ),
            )
        )


def get_datamodule_config(expt: Experiment):
    # default dataloaders for each dataset
    # can make this more complex if necessary
    default_mapping = {
        "acdc": "acdc_nnunet",
        "brats19_lhgg": "brats19_lhgg_nnunet",
        "covid_gonzalez": "covid_nnunet",
        "kits23": "kits23_nnunet",
        "mnms": "mnms_nnunet",
        "prostate_gonzalez": "prostate_nnunet",
        "simple_fets22_corrupted": "simple_fets22_corrupted",
        "retina": "retina_nnunet",
        "retouch_cirrus": "retouch_cirrus_nnunet",
        "retouch_spectralis": "retouch_spectralis_nnunet",
        "retouch_topcon": "retouch_topcon_nnunet",
        "mvseg23": "mvseg23_nnunet",
        "octa500": "octa500_nnunet",
    }
    if expt.csf_image is not None and expt.csf_image.endswith("quality_regression"):
        return "quality_regression"
    if expt.csf_image is not None and expt.csf_image.lower().startswith("vae"):
        return "vae"
    if expt.csf_aggregation is not None:
        if expt.csf_aggregation == "all_simple":
            return "simple_agg"
        elif "radiomics" in expt.csf_aggregation or "heuristic" in expt.csf_aggregation:
            return "heuristic_radiomics"
    return default_mapping[expt.dataset]
