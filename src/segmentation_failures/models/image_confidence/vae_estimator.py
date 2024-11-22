"""Adapted from David Zimmerer's code"""

from pathlib import Path

import monai
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as dist
import torch.nn as nn
from loguru import logger

# from segmentation_failures.networks.vae.utils import ConvUpsample, Conv3DUpsample
from omegaconf import ListConfig

from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics_and_info,
)
from segmentation_failures.networks.vae import VAE, VAE3d
from segmentation_failures.utils.dice_bce_loss import DiceBCEloss


class SimpleVAEmodule(pl.LightningModule):
    def __init__(
        self,
        img_dim: int,
        img_size: int | list[int],
        img_channels: int,
        seg_channels: int,
        z_dim: int,
        model_h_size: list[int],
        beta=0.01,
        recon_loss_img="l1",
        recon_loss_seg="bce",
        normalization_op="instance",
        lr=1e-4,
        log_n_samples=0,
        log_train_recons=False,
        log_val_recons=True,
        **network_kwargs,
    ):
        super(SimpleVAEmodule, self).__init__()

        self.save_hyperparameters()
        if isinstance(img_size, int):
            img_size = [img_size] * img_dim
        self.spatial_size = img_size
        self.img_channels = img_channels
        self.seg_channels = seg_channels
        input_shape = (img_channels + seg_channels, *img_size)  # without batch dimension
        if isinstance(model_h_size, ListConfig):
            model_h_size = list(model_h_size)
        self.beta = beta

        if img_dim == 2:
            vae_cls = VAE
        elif img_dim == 3:
            vae_cls = VAE3d
        else:
            raise ValueError(f"Image dimension must be 2 or 3, but is {img_dim}")
        if normalization_op == "batch":
            normalization_op = nn.BatchNorm2d if img_dim == 2 else nn.BatchNorm3d
        elif normalization_op == "instance":
            normalization_op = nn.InstanceNorm2d if img_dim == 2 else nn.InstanceNorm3d
        else:
            normalization_op = None

        self.vae = vae_cls(
            input_size=input_shape,
            z_dim=z_dim,
            h_sizes=model_h_size,
            normalization_op=normalization_op,
            **network_kwargs,
        )
        # self.best_img_pr, self.bext_pixel_pr = 0, 0
        self.recon_loss_img = recon_loss_img
        if recon_loss_seg == "bce":
            self.recon_loss_seg = nn.BCEWithLogitsLoss(reduction="none")
        elif recon_loss_seg == "dicebce":
            self.recon_loss_seg = DiceBCEloss({}, {}, use_ignore_label=False, reduction="none")
        elif recon_loss_seg == "dice":
            self.recon_loss_seg = DiceBCEloss(
                {}, {}, use_ignore_label=False, weight_ce=0, reduction="none"
            )
        else:
            raise ValueError(f"Unknown segmentation reconstruction loss {recon_loss_seg}")
        if recon_loss_img == "l1":
            self.recon_loss_img = nn.L1Loss(reduction="none")
        elif recon_loss_img == "l2":
            self.recon_loss_img = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss {self.recon_loss_img}")
        self.validation_buffer = []

    def make_input_from_batch(self, batch):
        # Depending on the module configuration, use img and/or seg as inputs to the VAE
        img_input, seg_input = torch.empty(0, device=self.device), torch.empty(
            0, device=self.device
        )
        if self.img_channels > 0:
            img_input = batch["data"]
            if isinstance(img_input, monai.data.MetaTensor):
                img_input = img_input.as_tensor()
            assert img_input.shape[1] == self.img_channels
        if self.seg_channels > 0:
            if self.trainer.state.fn in ["test", "predict"]:
                seg_input = batch["pred"]
            else:
                seg_input = batch["target"]
            if isinstance(seg_input, monai.data.MetaTensor):
                seg_input = seg_input.as_tensor()
            assert (
                seg_input.shape[1] == self.seg_channels
            ), f"{seg_input.shape[1]} != {self.seg_channels}"
        return torch.cat([img_input, seg_input.float()], dim=1)

    def trainval_step(self, batch, log_images=False, log_images_name="VAE orig_recon"):
        vae_input = self.make_input_from_batch(batch)
        x_rec_vae, z_dist = self.vae(vae_input, sample=self.beta != 0.0)
        kl_loss = 0
        if self.beta > 0:
            kl_loss = self.kl_loss_fn(z_dist)
        rec_loss_vae = self.rec_loss_fn(x_rec_vae, vae_input)
        loss_vae = kl_loss * self.beta + rec_loss_vae

        if log_images:
            self.tensorboard_log_images(log_images_name, x_rec_vae.detach(), vae_input.detach())

        logs = {
            "recon_loss": rec_loss_vae,
            "kl_loss": kl_loss,
            "loss": loss_vae,
        }
        return loss_vae, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.trainval_step(
            batch,
            log_images=self.hparams.log_train_recons
            and batch_idx % 10 == 0
            and self.current_epoch % 10 == 0,
            log_images_name="VAE train_recon",
        )
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.state.fn == "validate":
            if "pred" not in batch:
                logger.warning("No prediction in batch, skipping failure detection validation")
            else:
                self.fd_validation_step(batch, batch_idx)
        loss, logs = self.trainval_step(
            batch,
            log_images=self.hparams.log_val_recons and batch_idx == 0,
            log_images_name="VAE val_recon",
        )
        self.log_dict({f"val_{k}_epoch": v for k, v in logs.items()})
        return loss

    def fd_validation_step(self, batch, batch_idx):
        # include background means here "use all classes".
        # The background channel should have been removed by the dataloader
        metric_obj = get_metrics_and_info("dice", include_background=True)[0]["dice"]
        # compute confidence scores -> like testing
        test_vae_results = self.test_step(batch, batch_idx)
        confid_scores = test_vae_results["confidence"]["elbo"]
        # compute Dice(target, prediction)
        dice_scores = metric_obj(
            y=batch["target"], y_pred=batch["pred"]
        )  # these are class-wise dice scores
        mean_dice_scores = dice_scores.mean(dim=1)
        self.validation_buffer.append(
            {"confidence": confid_scores, "pred_score": mean_dice_scores}
        )

    def fd_validation_epoch_end(self):
        if len(self.validation_buffer) == 0:
            logger.warning("Validation buffer is empty, skipping failure detection validation")
            return
        # combine the values from the validation buffer into arrays for confidence and pred_score
        confid_scores = torch.cat([x["confidence"] for x in self.validation_buffer]).cpu().numpy()
        pred_scores = torch.cat([x["pred_score"] for x in self.validation_buffer]).cpu().numpy()
        # fit a linear regression model and log some metrics using scikit-learn
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        confid_scores = confid_scores.reshape(-1, 1)
        pred_scores = pred_scores.reshape(-1)
        ols_model = LinearRegression()
        scores = cross_val_score(
            ols_model,
            confid_scores,
            pred_scores,
            cv=LeaveOneOut(),
            scoring="neg_mean_absolute_error",
        )
        self.log_dict({"val_mae_linear_regression": -scores.mean()})

    def on_validation_epoch_end(self) -> None:
        if self.trainer.state.fn == "validate":
            self.fd_validation_epoch_end()
        if self.hparams.log_n_samples > 0:
            for _ in range(self.hparams.log_n_samples):
                vae_samples = self.vae.generate_samples(1, device=self.device)
                self.tensorboard_log_images("VAE samples", vae_samples.detach())

    def tensorboard_log_images(self, name, output_tensor, input_tensor=None):
        for tb_logger in self.loggers:
            if isinstance(tb_logger, pl.loggers.tensorboard.TensorBoardLogger):
                break
        if self.img_channels > 0:
            for ch_idx in range(self.img_channels):
                # image channels are at the front
                out_slice = extract_slice_for_visualization(output_tensor[:, ch_idx].unsqueeze(1))
                image_name = f"{name} (img channel {ch_idx})"
                if input_tensor is not None:
                    in_slice = extract_slice_for_visualization(
                        input_tensor[:, ch_idx].unsqueeze(1)
                    )
                    out_slice = torch.cat([in_slice, out_slice], dim=2)
                tb_logger.experiment.add_images(image_name, out_slice, self.global_step)

        if self.seg_channels > 0:
            # sigmoid because BCEWithLogitsLoss is used
            seg_output = torch.sigmoid(output_tensor[:, -self.seg_channels :])
            for ch_idx in range(self.seg_channels):
                out_slice = extract_slice_for_visualization(
                    seg_output[:, ch_idx].unsqueeze(1), normalize=False
                )
                img_name = f"{name} (masks class {ch_idx})"
                if input_tensor is not None:
                    in_slice = extract_slice_for_visualization(
                        input_tensor[:, -self.seg_channels + ch_idx].unsqueeze(1), normalize=False
                    )
                    out_slice = torch.cat([in_slice, out_slice], dim=2)
                tb_logger.experiment.add_images(img_name, out_slice, self.global_step)
        # if input_tensor is not None:
        #     # also save the raw data to disk
        #     out_dir = Path(tb_logger.log_dir) / "reconstructions"
        #     out_dir.mkdir(exist_ok=True)
        #     self.save_raw_reconstructions(
        #         input_tensor, output_tensor, out_dir / f"{name}_{self.global_step}"
        #     )

    def save_raw_reconstructions(self, input_tensor, output_tensor, save_path: Path):
        out_dict = {}
        out_dict["img_gen"] = output_tensor.detach().cpu().numpy()
        out_dict["img_orig"] = input_tensor.detach().cpu().numpy()
        if len(self.spatial_size) == 3:
            # to save disk space, only save the middle slice
            slice_idx = self.spatial_size[0] // 2
            out_dict["img_gen"] = out_dict["img_gen"][:, :, slice_idx]
            out_dict["img_orig"] = out_dict["img_orig"][:, :, slice_idx]
        np.savez_compressed(save_path, **out_dict)

    def test_step(self, batch, batch_idx):
        # NOTE this is failure detection testing, not VAE testing as in the validation_step
        vae_input = self.make_input_from_batch(batch)
        x_rec_vae, z_dist = self.vae(vae_input, sample=self.beta != 0.0)
        kl_loss = 0
        if self.beta > 0:
            kl_loss = self.kl_loss_fn(z_dist, sum_samples=False)
        rec_loss_vae = self.rec_loss_fn(x_rec_vae, vae_input, sum_samples=False)
        loss_vae = kl_loss * self.beta + rec_loss_vae
        return {
            "prediction": batch["pred"],
            "confidence": {"elbo": -loss_vae.detach()},
            "confidence_pixel": None,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def kl_loss_fn(self, z_post, sum_samples=True):
        z_prior = dist.Normal(0, 1.0)
        kl_div = dist.kl_divergence(z_post, z_prior)
        avg_dims = list(range(1, 2 + len(self.spatial_size)))  # CHW[D]
        if sum_samples:
            avg_dims = [0] + avg_dims  # BCHW[D]
        kl_div = torch.mean(kl_div, dim=avg_dims)
        return kl_div

    def rec_loss_fn(self, recon_x, x, sum_samples=True):
        recon_img, recon_seg = torch.split(recon_x, [self.img_channels, self.seg_channels], dim=1)
        input_img, input_seg = torch.split(x, [self.img_channels, self.seg_channels], dim=1)
        avg_dims = list(range(1, 2 + len(self.spatial_size)))  # CHW[D]
        if sum_samples:
            avg_dims = [0] + avg_dims  # BCHW[D]
        loss_img = 0
        if self.img_channels > 0:
            loss_img = self.recon_loss_img(recon_img, input_img)
            loss_img = torch.mean(loss_img, dim=avg_dims)
        loss_seg = 0
        if self.seg_channels > 0:
            # NOTE multi-class segmentation is currently not explicitly modeled
            loss_seg = self.recon_loss_seg(recon_seg, input_seg)
            loss_seg = torch.mean(loss_seg, dim=avg_dims)
        return loss_img + loss_seg


# Under construction
class IterativeSurrogateVAEmodule(SimpleVAEmodule):
    def __init__(
        self,
        surrogate_lr: float = 1e-3,
        quality_metric="generalized_dice",
        convergence_thresh: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.seg_channels == 0 or self.img_channels == 0:
            raise ValueError("IterativeSurrogateVAEmodel requires image AND segmentation channels")
        self.surrogate_lr = surrogate_lr
        metric_dict, metric_infos = get_metrics_and_info(quality_metric, include_background=True)
        self.quality_metric = metric_dict[quality_metric]
        self.quality_higher_better = metric_infos[quality_metric].higher_better
        self.convergence_thresh = convergence_thresh
        self.MAX_ITER = 100

    def test_step(self, batch, batch_idx):
        # NOTE this is failure detection testing, not VAE testing as in the validation_step
        vae_input = self.make_input_from_batch(batch)
        mu_z, _ = self.vae.encode(vae_input)
        vae_output = self.vae.decode(mu_z)
        # TODO is it possible to optimize the samples jointly?
        confidence = []
        torch.set_grad_enabled(True)
        self.vae.requires_grad_(False)  # don't need gradients for the model parameters
        for sample_idx in range(vae_input.shape[0]):
            z_surr = mu_z[sample_idx].clone().unsqueeze(0)  # BZHW[D]
            z_surr.requires_grad = True
            surrogate_optimizer = torch.optim.Adam([z_surr], lr=self.surrogate_lr)
            ema_decay = 0.5
            ema_z = torch.randn_like(z_surr, requires_grad=False)
            for _ in range(self.MAX_ITER):
                x_orig = vae_input[sample_idx].unsqueeze(0)
                ema_z = ema_decay * ema_z + (1 - ema_decay) * z_surr.detach().clone()
                # perform a gradient step
                surrogate_optimizer.zero_grad()
                x_rec = self.vae.decode(z_surr)
                # TODO the objective of the surrogate optimization is not clear from Wang et al.
                loss = self.rec_loss_fn(x_rec, x_orig, sum_samples=True)
                loss.backward()
                surrogate_optimizer.step()
                # check convergence
                # TODO this is very primitive so far; Wange et al don't specify details in the paper
                if torch.norm((ema_z - z_surr).flatten(), p=2) < self.convergence_thresh:
                    break
            # compare x_rec vae_output
            # convert to mask first (needs to be adapted to the loss! I use BCE currently)
            vae_output = (vae_output > 0).to(dtype=int)
            x_rec = (x_rec.detach() > 0).to(dtype=int)
            # TODO maybe log the two masks for debugging
            quality_pred = self.estimate_quality(
                x_rec.detach(), vae_output[sample_idx].unsqueeze(0)
            )
            assert quality_pred.nelement() == 1
            confidence.append(quality_pred)
        torch.set_grad_enabled(False)
        confidence = torch.concatenate(confidence)
        return {
            "prediction": batch["pred"],
            "confidence": {"surrogate_quality": confidence},
            "confidence_pixel": None,
        }

    def estimate_quality(self, x_recon, x_surrog):
        # I think Wang et al. just compute the dice score between the two masks (ignoring images)
        _, seg_recon = torch.split(x_recon, [self.img_channels, self.seg_channels], dim=1)
        _, seg_surrog = torch.split(x_surrog, [self.img_channels, self.seg_channels], dim=1)
        result = self.quality_metric(seg_recon, seg_surrog)
        if not self.quality_higher_better:
            result *= -1
        return result


def extract_slice_for_visualization(tensor: torch.Tensor, normalize=True):
    if tensor.ndim == 5:
        # 3d case: take middle slice
        slice_dim = 2  # BCDHW
        tensor = tensor.select(dim=slice_dim, index=tensor.shape[slice_dim] // 2)
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # to rgb
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, *([1] * (tensor.ndim - 2)))
    return tensor
