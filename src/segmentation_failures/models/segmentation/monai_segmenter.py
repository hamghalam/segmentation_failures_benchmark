import pytorch_lightning as pl
import torch
from loguru import logger
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType


class UNet_segmenter(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay=1e-5,
        sw_patch_size: list[int] | None = None,
        sw_batch_size: int = 1,
        sw_overlap: float = 0.5,
    ):
        """Basic segmentation module with configurable architecture.

        Args:
            backbone (torch.nn.Module): Neural network to use for prediction.
            num_classes (int): number of classes in segmentation task
            lr (float, optional): learning rate of AdamW optimizer. Defaults to 1e-3.
            weight_decay (float, optional): weight decay of AdamW optimizer. Defaults to 1e-5.
            sw_patch_size: Can be None if no sliding window should be used (e.g. 2D)
            sw_batch_size (int): number of patches per batch for sliding window inference
            sw_overlap (float): fractional overlap between sw patches.
        """
        super().__init__()
        self.save_hyperparameters(ignore="backbone")

        self.model = backbone
        if isinstance(sw_overlap, float) and sw_patch_size is not None:
            sw_overlap = tuple(int(sw_overlap * ps) for ps in sw_patch_size)
        self.sw_patch_size = sw_patch_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])
        # Cases/classes with missing GT are set to nan
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean_batch", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        """Inference forward pass

        Args:
            x (torch.Tensor): Shape BCHW[D]

        Returns:
            prediction (usually logits)
        """
        # For 2D, for example, don't do sliding window inference
        inferer = SimpleInferer()
        if x.ndim == 5 or self.sw_patch_size is not None:
            inferer = SlidingWindowInferer(
                roi_size=self.sw_patch_size,
                sw_batch_size=self.sw_batch_size,
                overlap=self.sw_overlap,
                mode="gaussian",
            )
        img_size = x.shape[2:]
        expected_size = getattr(self.model, "spatial_dims", img_size)
        if hasattr(expected_size, "__len__"):
            expected_size = len(expected_size)
        if expected_size != len(img_size):
            raise ValueError(
                f"Got an input image with {x.dim()}, but expected "
                f"{2 + expected_size} (batch + channel + spatial dims)"
            )
        # NOTE The image shape may have to be divisible by some power of 2 for some networks.
        # Can't check this here, need to pad correctly
        return inferer(x, network=self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        patches, labels = batch["data"], batch["target"]
        output = self.model(patches)
        loss = self.loss_function(output, labels)
        self.log("train_loss", loss, on_epoch=True, batch_size=len(batch["data"]))
        return {"loss": loss, "prediction": output.detach()}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["data"], batch["target"]
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        # # there's some error in decollate_batch with batch_size...
        # outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        # labels = [self.post_label(i) for i in decollate_batch(labels)]
        dice_scores = self.dice_metric(
            y_pred=[self.post_pred(i) for i in torch.unbind(outputs, dim=0)],
            y=[self.post_label(i) for i in torch.unbind(labels, dim=0)],
        )
        self.log("val_loss", loss, on_step=True, batch_size=len(batch["data"]))
        return {"loss": loss, "prediction": outputs.detach(), "dice": dice_scores}

    def on_validation_epoch_end(self):
        mean_val_dice_per_class = self.dice_metric.aggregate()
        avg_val_dice = mean_val_dice_per_class.mean().item()
        self.dice_metric.reset()
        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            self.best_val_epoch = self.current_epoch
        logger.info(f"current mean dice (per class): {mean_val_dice_per_class}")
        logger.info(
            f"best mean dice (avg. over classes): {self.best_val_dice:.4f} at epoch: {self.best_val_epoch}"
        )
        self.log_dict(
            {f"val_dice_class{i}": dice.item() for i, dice in enumerate(mean_val_dice_per_class)}
        )

    def test_step(self, batch, batch_idx):
        images, labels = batch["data"], batch["target"]
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.dice_metric(
            y_pred=[self.post_pred(i) for i in torch.unbind(outputs, dim=0)],
            y=[self.post_label(i) for i in torch.unbind(labels, dim=0)],
        )
        self.log("test_loss", loss, on_step=True, batch_size=len(batch["data"]))
        return {"prediction": outputs}

    def on_test_epoch_end(self):
        mean_dice_per_class = self.dice_metric.aggregate()
        self.dice_metric.reset()
        self.log_dict(
            {f"test_dice_class{i}": dice.item() for i, dice in enumerate(mean_dice_per_class)}
        )
