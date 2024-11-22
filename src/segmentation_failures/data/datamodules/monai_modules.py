import random
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import monai.transforms as trf
import pytorch_lightning as pl
from loguru import logger
from monai.data import CacheDataset, DataLoader

from segmentation_failures.utils.io import load_json


class MonaiBaseModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        dataset_id: str,
        batch_size: int = 4,
        batch_size_inference: int = 1,
        patch_size: tuple[int] | list[int] = None,
        val_size: float = 0.2,
        fold: int | None = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_num: Union[float, int] = 1.0,
        domain_mapping_json: str = "domain_mapping_00.json",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.img_dims = None
        self.dataset_id = dataset_id
        self.test_case_ids = []
        self.train_data_dir = Path(train_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.dataset_json = load_json(self.train_data_dir / "dataset.json")
        if domain_mapping_json is not None:
            self.domain_mapping = load_json(self.test_data_dir / domain_mapping_json)

    @abstractmethod
    def get_transforms(self):
        pass

    @abstractmethod
    def read_img(self, img_path: str):
        # This is not used so far
        pass

    @abstractmethod
    def read_seg(self, seg_path: str):
        pass

    def setup_train(self):
        logger.info(f"Loading data from {self.train_data_dir}")
        # set up the correct data path
        data_dicts = []
        for label_path in (self.train_data_dir / "labelsTr").glob("*.nii.gz"):
            case_id = label_path.name.removesuffix(".nii.gz")
            img_list = []
            for mod_idx in self.dataset_json["channel_names"]:
                img_path = (
                    self.train_data_dir / "imagesTr" / f"{case_id}_{int(mod_idx):04d}.nii.gz"
                )
                img_list.append(img_path)
            data_dicts.append(
                {
                    "target": label_path,
                    "data": img_list,
                    "keys": case_id,
                }
            )

        train_files = []
        val_files = []
        if self.hparams.fold is not None:
            # split_path = task_data_dir / "trainval_splits" / f"fold_{self.hparams.fold}.json"
            # logger.info(f"Reading train/val split from file {split_path}.")
            split_path = self.train_data_dir / "splits_final.json"
            # json with train and val keys, each containing a list of subjects (exclusive)
            split_dict = load_json(split_path)[self.hparams.fold]
            for dd in data_dicts:
                case_id = dd["target"].name.removesuffix(".nii.gz")
                if case_id in split_dict["train"]:
                    train_files.append(dd)
                elif case_id in split_dict["val"]:
                    val_files.append(dd)
                else:
                    raise RuntimeError(f"Case {case_id} not found in the split file")
        else:
            num_val = int(self.hparams.val_size * len(data_dicts))
            random.shuffle(data_dicts)
            train_files, val_files = data_dicts[:-num_val], data_dicts[-num_val:]
        logger.info(
            f"Total #cases = {len(data_dicts)}. Thereof {len(train_files)}/{len(val_files)} for training/validation"
        )

        # define the data transforms
        train_transforms, val_transforms = self.get_transforms()

        cache_num = self.hparams.cache_num
        if isinstance(cache_num, float):
            cache_num = cache_num * (len(train_files) + len(val_files))
        # not sure what is the optimal split (wrt training runtime); could also cache only train samples
        cache_num_train = int(cache_num * len(train_files) / (len(train_files) + len(val_files)))
        cache_num_valid = int(cache_num * len(val_files) / (len(train_files) + len(val_files)))
        self.train_data = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_num=cache_num_train,
            num_workers=4,
        )
        self.valid_data = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=cache_num_valid,
            num_workers=4,
        )

    def setup_test(self):
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"No data was found here: {self.test_data_dir}")
        logger.info(f"Loading data from {self.test_data_dir}")
        data_dicts = self.get_test_data_dicts()

        # define the data transforms
        _, test_transforms = self.get_transforms()

        cache_num = self.hparams.cache_num
        if isinstance(cache_num, float):
            cache_num = int(cache_num * len(data_dicts))
        self.test_data = CacheDataset(
            data=data_dicts,
            transform=test_transforms,
            cache_num=cache_num,
            num_workers=4,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "validate" or stage is None:
            self.setup_train()
        elif stage == "test":
            self.setup_test()
        else:
            raise ValueError(f"stage must be in fit/test. Got {stage}")

    def train_dataloader(self):
        if self.train_data is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        if self.valid_data is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        return DataLoader(
            self.valid_data,
            shuffle=False,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        if self.test_data is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        logger.warning(
            "This dataloader is identical to test_dataloader "
            "and was added just for getting rid of a warning."
        )
        return self.test_dataloader()

    def get_test_data_dicts(self):
        # set up the correct data path
        img_dir = self.test_data_dir / "imagesTs"
        lab_dir = self.test_data_dir / "labelsTs"
        if not img_dir.exists():
            raise FileNotFoundError(f"No data was found here: {img_dir}")

        data_dicts = []
        for label_path in lab_dir.glob("*.nii.gz"):
            case_id = label_path.name.split(".")[0]
            img_list = []
            for mod_idx in self.dataset_json["channel_names"]:
                img_path = img_dir / f"{case_id}_{int(mod_idx):04d}.nii.gz"
                img_list.append(img_path)
            curr_dict = {
                "keys": case_id,
                "target": label_path,
                "data": img_list,
            }
            if self.domain_mapping:
                curr_dict["domain_label"] = self.domain_mapping[case_id]
            data_dicts.append(curr_dict)
            self.test_case_ids.append(case_id)
        return data_dicts

    def load_case_id(self, case_id: str, dataset="train"):
        if dataset == "train" or dataset == "valid":
            raise NotImplementedError
        elif dataset == "test":
            ds = self.test_data
        else:
            raise ValueError

        if ds is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying data."
            )
        idx = self.test_case_ids.index(case_id)
        return ds[idx]


class SimpleBraTS(MonaiBaseModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        dataset_id: str,
        batch_size: int = 4,
        batch_size_inference: int = 1,
        patch_size: tuple[int] | list[int] = None,
        spacing: tuple[float] | list[float] = None,
        val_size: float = 0.2,
        fold: int | None = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_num: bool = True,
    ) -> None:
        super().__init__(
            train_data_dir=train_data_dir,
            test_data_dir=test_data_dir,
            dataset_id=dataset_id,
            batch_size=batch_size,
            batch_size_inference=batch_size_inference,
            val_size=val_size,
            fold=fold,
            num_workers=num_workers,
            pin_memory=pin_memory,
            cache_num=cache_num,
        )
        self.save_hyperparameters()

    def get_transforms(self):
        all_keys = ["data", "target"]
        train_transforms = [
            trf.LoadImaged(keys=all_keys),
            trf.EnsureChannelFirstd(keys=all_keys),
            trf.NormalizeIntensityd(keys="data", channel_wise=True),
            trf.EnsureTyped(keys=all_keys),
            trf.RandFlipd(keys=all_keys, spatial_axis=[0], prob=0.5),
            trf.RandFlipd(keys=all_keys, spatial_axis=[1], prob=0.5),
        ]

        val_transforms = [
            trf.LoadImaged(keys=all_keys),
            trf.EnsureChannelFirstd(keys=all_keys),
            trf.NormalizeIntensityd(keys="data", channel_wise=True),
            trf.EnsureTyped(keys=all_keys),
        ]
        return trf.Compose(train_transforms), trf.Compose(val_transforms)

    def read_img(self, img_paths: str | list[str]):
        img_tensor = trf.LoadImage()(img_paths)
        properties = {"spacing": (1.0, 1.0)}
        return trf.ToNumpy()(img_tensor), properties

    def read_seg(self, seg_path: str):
        img_tensor, img_header = trf.LoadImage(ensure_channel_first=True, image_only=False)(
            seg_path
        )
        # Not sure if this works for all data types, but nifti should be fine.
        properties = {"spacing": img_header["pixdim"][1 : img_header["dim"][0] + 1]}
        return trf.ToNumpy()(img_tensor), properties
