from pathlib import Path

import monai
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Callback
from torchvision.transforms import ToPILImage

from segmentation_failures.utils.visualization import make_image_mask_grid


class BatchVisualization(Callback):
    def __init__(
        self,
        num_classes: int,
        log_dir=None,
        every_n_steps=250,
        max_num_images=16,
    ):
        self.num_classes = num_classes
        self.log_dir = log_dir
        self.every_n_batches = every_n_steps
        self.max_num_images = max_num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        # Log a subset of the training images and their labels as images to tensorboard
        # if batch_idx > 0:
        #     # Log only once per epoch
        if trainer.global_step % self.every_n_batches == 0:
            self.log_batch(trainer, batch, outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking and batch_idx == 0:
            self.log_batch(trainer, batch, outputs, mode="val")

    def log_batch(self, trainer, batch, outputs, mode="train"):
        # optional: log to tensorboard
        for tb_logger in trainer.loggers:
            if isinstance(tb_logger, pl.loggers.tensorboard.TensorBoardLogger):
                break
        self._log_batch(batch, outputs, mode, step=trainer.global_step, tb_logger=tb_logger)

    def _log_batch(self, batch, outputs=None, mode="train", step=0, tb_logger=None):
        images = batch.get("data")
        labels = batch.get("target")
        reverse_spatial = "properties" in batch  # nnunet dataloader
        if images is None and labels is None:
            logger.warning("No images or labels found in batch. Skipping logging.")
            return
        elif images is None or not isinstance(images, torch.Tensor):
            images = torch.zeros_like(labels)
            logger.warning("No images found in batch. Logging only labels.")
        elif labels is None or not isinstance(images, torch.Tensor):
            labels = torch.zeros_like(images)
            logger.warning("No labels found in batch. Logging only images.")
        if isinstance(labels, list):
            labels = labels[0]  # deep supervision
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        num_spatial = len(images.shape[2:])
        if isinstance(images, monai.data.MetaTensor):
            images = images.as_tensor()
        if isinstance(labels, monai.data.MetaTensor):
            labels = labels.as_tensor()
        exclusive_labels = labels.shape[1] == 1
        if exclusive_labels:
            # exclusive labels
            # FIXME the maximum in here is due to bad configuration files: see num_fg_classes vs num_classes for brats or kits
            labels = F.one_hot(
                labels.squeeze(1).to(torch.long),
                num_classes=max(self.num_classes, labels.max() + 1),
            )
            labels = labels.permute(0, -1, *range(1, num_spatial + 1))
        slice_indices = None
        if num_spatial == 3:
            # select slice with largest foreground fraction; BCHWD -> B
            sum_dims = (1, 3, 4) if reverse_spatial else (1, 2, 3)
            slice_indices = torch.argmax(
                torch.sum(labels[:, int(exclusive_labels) :], dim=sum_dims), dim=1
            ).tolist()
        rgb_image = make_image_mask_grid(
            image_batch=images,
            mask_list=[labels],
            max_images=self.max_num_images,
            slice_idx=slice_indices,
            slice_dim=0 if reverse_spatial else 2,
        )
        if self.log_dir is not None:
            fpath = Path(self.log_dir) / f"{mode}_batch_{step}.png"
            # save tensor as png
            img = ToPILImage()(rgb_image)
            img.save(fpath)
        if tb_logger is not None:
            tb_logger.experiment.add_image(f"{mode}_batch", rgb_image, step, dataformats="CHW")
