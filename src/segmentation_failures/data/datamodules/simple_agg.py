import os
from pathlib import Path
from typing import Optional

import monai
import monai.transforms as trf
import pytorch_lightning as pl
import torch
from loguru import logger
from monai.data import CacheDataset, DataLoader

from segmentation_failures.data.datamodules.additional_readers import TiffReader
from segmentation_failures.data.datamodules.quality_regression import (
    SegMetricTargetComputation,
)
from segmentation_failures.data.datamodules.vae import check_if_files_exist
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics_and_info,
)
from segmentation_failures.utils.data import get_dataset_dir, load_dataset_json
from segmentation_failures.utils.io import load_json


class SimpleAggDataModule(pl.LightningDataModule):
    # So far only for testing!
    def __init__(
        self,
        dataset_id: int,
        fold: int,
        prediction_dir: str,
        prediction_samples_dir: str = None,
        confid_dir: str = None,
        confid_name: str = None,
        metric_targets: str | list[str] = None,
        test_data_root: str = None,
        num_workers: int | None = None,
        pin_memory: bool = False,
        domain_mapping: int = 0,
        include_background: bool = False,
        expt_group="default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.domain_mapping = domain_mapping
        if isinstance(dataset_id, str):
            dataset_id = int(dataset_id)
        self.dataset_id = dataset_id
        self.num_workers = num_workers
        if test_data_root is None:
            test_data_root = os.environ["TESTDATA_ROOT_DIR"]
        # hard-coded sh*t, but whatever
        if dataset_id == 500:
            self.train_data_root = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"])
            split_path = self.train_data_root / "splits_final.json"
        else:
            self.train_data_root = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"])
            split_path = (
                get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"])
                / "splits_final.json"
            )
        self.train_val_split = load_json(split_path)[fold]
        self.test_data_root = get_dataset_dir(dataset_id, test_data_root)
        self.prediction_dir = Path(prediction_dir)
        if not Path(prediction_dir).is_absolute():
            self.prediction_dir = (
                get_dataset_dir(dataset_id, os.environ["SEGFAIL_AUXDATA"])
                / expt_group
                / prediction_dir
                / "predictions"
            )
        self.prediction_samples_dir = Path(prediction_samples_dir)
        self.confid_dir = Path(confid_dir)
        if not self.confid_dir.is_absolute():
            self.confid_dir = (
                get_dataset_dir(dataset_id, os.environ["SEGFAIL_AUXDATA"])
                / expt_group
                / confid_dir
                / "confidence_maps"
            )
        self.confid_name = confid_name
        self.dataset_test: CacheDataset = None
        self._dataloader_test = None
        # -> adapted from quality regression module
        if isinstance(metric_targets, str):
            metric_targets = [metric_targets]
        if metric_targets is None:
            # not strictly necessary for testing
            self.metric_targets, metric_infos = None, {}
        else:
            self.metric_targets, metric_infos = get_metrics_and_info(
                metric_targets,
                include_background=include_background,
            )
        self.metric_target_names = []
        self.metric_higher_better = []
        self.dataset_json = load_dataset_json(dataset_id)
        for metric_name, metric_info in metric_infos.items():
            if metric_info.classwise:
                num_classes = len(self.dataset_json["labels"]) - (not include_background)
                self.metric_target_names += [f"{metric_name}_{i}" for i in range(num_classes)]
                self.metric_higher_better += [metric_info.higher_better] * num_classes
            else:
                self.metric_target_names.append(metric_name)
                self.metric_higher_better.append(metric_info.higher_better)
        # <- copied from quality regression module

    def setup_train(self):
        # set up the correct data path
        train_dicts, val_dicts = self.get_train_data_dicts()
        has_confids = "confid" in train_dicts[0]
        has_pred_samples = "pred_samples" in train_dicts[0]
        logger.info(f"Found {len(train_dicts)}/{len(val_dicts)} cases for training/validation")
        load_image_kwargs = {}
        file_type = ".".join(train_dicts[0]["pred"].split(".")[1:])
        if file_type in ["tiff", "tif"]:
            # special case: tiff reader
            load_image_kwargs["reader"] = TiffReader(rgb=True)
        # get data transforms
        transforms = get_transforms(
            pred_key="pred",
            pred_sample_keys=["pred_samples"] if has_pred_samples else None,
            confid_keys=["confid"] if has_confids else None,
            seg_metrics=self.metric_targets,
            labels_or_regions_defs=self.dataset_json["labels"],
            load_image_kwargs=load_image_kwargs,
        )
        self.dataset_train = CacheDataset(
            data=train_dicts,
            transform=transforms,
            cache_num=0,
            num_workers=self.num_workers,
        )
        self.dataset_val = CacheDataset(
            data=val_dicts,
            transform=transforms,
            cache_num=0,
            num_workers=self.num_workers,
        )

    def setup_test(self):
        data_dicts = self.get_test_data_dicts()
        has_confids = "confid" in data_dicts[0]
        has_pred_samples = "pred_samples" in data_dicts[0]
        load_image_kwargs = {}
        file_type = ".".join(data_dicts[0]["pred"].split(".")[1:])
        if file_type in ["tiff", "tif"]:
            # special case: tiff reader
            load_image_kwargs["reader"] = TiffReader(rgb=True)
        test_transforms = get_transforms(
            pred_key="pred",
            pred_sample_keys=["pred_samples"] if has_pred_samples else None,
            confid_keys=["confid"] if has_confids else None,
            labels_or_regions_defs=self.dataset_json["labels"],
            load_image_kwargs=load_image_kwargs,
        )
        self.dataset_test = CacheDataset(
            data_dicts,
            transform=test_transforms,
            cache_num=0,
            num_workers=self.num_workers,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit", "validate"] or stage is None:
            self.setup_train()
        elif stage == "test":
            self.setup_test()
        else:
            raise ValueError(f"stage must be fit/test/validate. Got {stage}")

    def get_train_data_dicts(self):
        pred_to_case_id_mapping = load_json(self.prediction_dir / "prediction_to_case_id.json")
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        pred_file_generator = self.prediction_dir.glob("*" + suffix)
        train_gt_dir = self.train_data_root / "labelsTr"
        train_files = []
        val_files = []

        for pred_file in pred_file_generator:
            case_id = pred_to_case_id_mapping[pred_file.name]
            label_file = str(train_gt_dir / f"{case_id}{suffix}")
            # add confidence
            curr_data_dict = {
                "keys": case_id,
                "target": label_file,
                "pred": str(pred_file),
                # optional:
                # confid, pred_samples
            }
            # add confidence maps if available
            if self.confid_dir.exists():
                all_confid_names = []
                curr_confids = []
                for subdir in self.confid_dir.iterdir():
                    if not subdir.is_dir() or (
                        self.confid_name is not None and subdir.name != self.confid_name
                    ):
                        continue
                    all_confid_names.append(subdir.name)
                    curr_confids.append(str(subdir / pred_file.name))
                if len(curr_confids) > 0:
                    curr_data_dict["confid"] = curr_confids
                    curr_data_dict["confid_names"] = all_confid_names
                else:
                    logger.warning(f"Found no confidence maps for {case_id}")
            # add prediction samples (ensemble) if available
            if self.prediction_samples_dir.exists():
                pred_samples_files = list(
                    self.prediction_samples_dir.glob(f"{case_id}_[0-9][0-9]{suffix}")
                )
                if len(pred_samples_files) == 0:
                    raise FileNotFoundError(f"Found no prediction samples: {pred_samples_files}")
                curr_data_dict["pred_samples"] = [str(x) for x in pred_samples_files]
            for k, v in curr_data_dict.items():
                if k in ["target", "pred", "confid"] and v is not None:
                    check_if_files_exist(v)
            # split
            if case_id in self.train_val_split["train"]:
                train_files.append(curr_data_dict)
            elif case_id in self.train_val_split["val"]:
                val_files.append(curr_data_dict)
            else:
                logger.warning(f"Case {case_id} not found in the split file")
        return train_files, val_files

    def get_test_data_dicts(self):
        data_dicts = []
        domain_mapping_path = (
            self.test_data_root / f"domain_mapping_{self.domain_mapping:02d}.json"
        )
        domain_mapping = None
        if domain_mapping_path.exists():
            domain_mapping = load_json(domain_mapping_path)
        test_gt_dir = self.test_data_root / "labelsTs"
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        for label_file in test_gt_dir.glob("*" + suffix):
            case_id = label_file.name.removesuffix(suffix)
            # add prediction
            pred_file = list(self.prediction_dir.glob(f"{case_id}.*"))
            if len(pred_file) == 0:
                raise FileNotFoundError(f"Found no predictions: {pred_file}")
            assert len(pred_file) == 1, f"Found multiple predictions: {pred_file}"
            pred_file = str(pred_file[0])
            curr_data_dict = {
                "keys": case_id,
                "target": str(label_file),
                "pred": pred_file,
                # optional:
                # confid
                # domain_label
            }
            # add confidence maps if available
            if self.confid_dir.exists():
                all_confid_names = []
                curr_confids = []
                for subdir in self.confid_dir.iterdir():
                    if not subdir.is_dir() or (
                        self.confid_name is not None and subdir.name != self.confid_name
                    ):
                        continue
                    all_confid_names.append(subdir.name)
                    confid_file = list(subdir.glob(f"{case_id}.*"))
                    assert (
                        len(confid_file) == 1
                    ), f"Expected one confidence map but got {len(confid_file)}"
                    curr_confids.append(str(confid_file[0]))
                if len(curr_confids) > 0:
                    curr_data_dict["confid"] = curr_confids
                    curr_data_dict["confid_names"] = all_confid_names
                else:
                    logger.warning(f"Found no confidence maps for {case_id}")
            # add prediction samples (ensemble) if available
            if self.prediction_samples_dir.exists():
                pred_samples_files = list(
                    self.prediction_samples_dir.glob(f"{case_id}_[0-9][0-9]{suffix}")
                )
                if len(pred_samples_files) == 0:
                    raise FileNotFoundError(f"Found no prediction samples: {pred_samples_files}")
                curr_data_dict["pred_samples"] = [str(x) for x in pred_samples_files]
            if domain_mapping is not None:
                curr_data_dict["domain_label"] = domain_mapping[case_id]
            for k, v in curr_data_dict.items():
                if k in ["target", "pred", "confid"] and v is not None:
                    check_if_files_exist(v)
            data_dicts.append(curr_data_dict)
        return data_dicts

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


def get_transforms(
    pred_key: str,
    pred_sample_keys: list[str] = None,
    confid_keys: list[str] = None,
    seg_metrics: dict[str, monai.metrics.Metric] = None,
    labels_or_regions_defs=None,
    load_image_kwargs=None,
):
    if confid_keys is None:
        confid_keys = []
    if pred_sample_keys is None:
        pred_sample_keys = []
    _load_image_kwargs = {
        "image_only": True,
        "ensure_channel_first": True,
    }
    if load_image_kwargs is not None:
        _load_image_kwargs.update(load_image_kwargs)
    load_image_trf = trf.LoadImaged(
        keys=["target", pred_key] + pred_sample_keys + confid_keys,
        **_load_image_kwargs,
    )
    # HACK: MONAI doesn't treat custom readers as documented in the API
    # I want to register the reader, but have the option to fall back to alternative readers (e.g. for numpy)
    load_image_trf._loader.auto_select = True
    # 1. load transforms
    transforms = [
        load_image_trf,
        trf.ToTensord(keys=["target", pred_key] + pred_sample_keys + confid_keys),
    ]
    if seg_metrics:
        transforms += [
            # could be precomputed, but for simplicity I imitate the quality regression module here
            SegMetricTargetComputation(
                pred_key, "target", seg_metrics, labels_or_regions_defs=labels_or_regions_defs
            ),
        ]
    transforms += [
        trf.CastToTyped(keys=["target", pred_key] + pred_sample_keys, dtype=torch.uint8),
    ]
    transforms = trf.Compose(transforms)
    return transforms
