from collections import defaultdict
from copy import copy

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures import resnet
from pytorch_lightning import LightningModule

from segmentation_failures.networks.dynunet import get_network as get_dynunet


# currently not used
def get_regression_network_resnet(img_dim, in_channels, num_outputs, network_name):
    net_cls = getattr(resnet, network_name)
    return net_cls(
        n_classes=num_outputs,
        n_input_channels=in_channels,
        input_dimension=img_dim,
    )


def get_regression_network_dynunet(
    img_dim, in_channels, num_outputs, voxel_spacing, img_size, blocks_per_stage=1
):
    # NOTE min_size is not enforced by this module; it's just used to determine the network architecture
    # use the dynamic unet for maximum flexibility
    dynunet = get_dynunet(
        out_channels=num_outputs,
        spatial_dims=img_dim,
        in_channels=in_channels,
        patch_size=img_size,
        spacings=voxel_spacing,
        res_block=True,
        blocks_per_stage=blocks_per_stage,
    )
    encoder = nn.Sequential(
        *([dynunet.input_block] + [layer for layer in dynunet.downsamples] + [dynunet.bottleneck])
    )
    final_num_channels = dynunet.filters[-1]
    # my best guess how this could be turned into a regression network
    quality_network = nn.Sequential(
        encoder,
        nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(final_num_channels, num_outputs),
    )
    return quality_network


class QualityRegressionNet(LightningModule):
    def __init__(
        self,
        output_names: list[str],  # metrics
        img_channels: int,
        img_dim: int,
        num_classes: int,
        voxel_spacing: tuple[float],
        img_size: tuple[int],
        confid_name: str = None,
        loss="l2",
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        cosine_annealing: bool = True,
        blocks_per_stage: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = img_channels + num_classes
        # the background class is not handled separately here; need to take care of that in the data loader
        self.output_names = output_names
        num_outputs = len(output_names)
        self.confid_name = confid_name
        if confid_name is not None:
            self.in_channels += 1
        if img_dim not in (2, 3):
            raise ValueError(f"Image dimension must be 2 or 3, but is {img_dim}")
        if img_dim == 2:
            self.quality_network = get_regression_network_resnet(
                img_dim, self.in_channels, num_outputs, "ResNet18"
            )
        else:
            self.quality_network = get_regression_network_dynunet(
                img_dim, self.in_channels, num_outputs, voxel_spacing, img_size, blocks_per_stage
            )
        if loss in ["mse", "l2"]:
            self.loss_fn = nn.MSELoss()
        elif loss in ["mae", "l1"]:
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function {loss}")
        # determine which class-wise metrics need to be averaged to get mean_metric
        self.metrics_to_average = defaultdict(list)
        for idx, name in enumerate(output_names):
            if name.split("_")[-1].isdigit():
                # class-wise metric format: {metric_name}_{class_idx}
                metric_name = "_".join(name.split("_")[:-1])
                self.metrics_to_average[metric_name].append(idx)

    def parse_batch(self, batch):
        image, seg = batch["data"], batch["pred"]
        # shape BCHW[D] and BKHW[D], respectively. Prediction must be one-hot encoded
        y_true = batch.get("metric_target", None)  # shape BM; misses for test set
        if self.confid_name is not None:
            pxl_confid = batch["confid"]  # shape B1HW[D]
            assert pxl_confid.shape[1] == 1
            x = torch.cat([image, seg, pxl_confid], dim=1)
        else:
            x = torch.cat([image, seg], dim=1)
        return x, y_true

    def training_step(self, batch, batch_idx):
        x, y_true = self.parse_batch(batch)
        y_pred = self.quality_network(x)  # shape BM
        loss = self.loss_fn(y_pred, y_true)
        self.log("train_loss", loss, batch_size=len(y_true))
        return {
            "loss": loss,
            "quality_true": y_true,
            "quality_pred": y_pred.detach(),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = self.parse_batch(batch)
        y_pred = self.quality_network(x)  # shape BM
        loss = self.loss_fn(y_pred, y_true)
        self.log("val_loss_epoch", loss, batch_size=len(y_true), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, _ = self.parse_batch(batch)
        segmentation = batch["pred"]  # one-hot
        quality_pred = self.quality_network(x)  # shape BM
        if quality_pred.ndim == 1:
            # special case: the dna.resnet squeezes the output, unfortunately
            quality_pred = quality_pred.reshape((segmentation.shape[0], -1))
        # Multiply quality with -1 to get confidence score for metrics that are lower-better
        # Maybe improve the self.trainer.datamodule.metric_higher_better solution later
        higher_better = (
            2
            * torch.tensor(self.trainer.datamodule.metric_higher_better).reshape(
                (1, quality_pred.shape[1])
            )
            - 1
        )
        confidence = higher_better.to(quality_pred.device) * quality_pred
        confid_names = copy(self.output_names)
        # add mean_metric to the output
        for metric_name, avg_idxs in self.metrics_to_average.items():
            confid_names.append(f"mean_{metric_name}")
            mean_confid = confidence[:, avg_idxs].mean(dim=1)
            confidence = torch.cat([confidence, mean_confid.unsqueeze(dim=1)], dim=1)
        return {
            "prediction": segmentation,
            "confidence": {name: confidence[:, i] for i, name in enumerate(confid_names)},
            "metric_prediction": quality_pred,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.quality_network.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.cosine_annealing:
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                ),
                "interval": "epoch",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer
