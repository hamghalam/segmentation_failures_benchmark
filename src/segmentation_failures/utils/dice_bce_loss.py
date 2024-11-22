"""
Adapted from nnunetv2
"""

import torch
import torch.nn as nn
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class MemoryEfficientSoftDiceLossWithReduction(MemoryEfficientSoftDiceLoss):
    # I added this class to allow for not reducing the dice loss in the batch dimension
    def __init__(self, reduction="none", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = reduction.lower()
        if reduction.lower() not in ["none", "mean", "sum", "mean_channels"]:
            raise ValueError("reduction must be None, 'none', 'mean' or 'sum'")

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = list(range(2, len(x.shape)))
        with torch.no_grad():
            if len(x.shape) != len(y.shape):
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        intersect = (
            (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        )
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        # if self.ddp and self.batch_dice:
        #     intersect = AllGatherGrad.apply(intersect).sum(0)
        #     sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
        #     sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            if self.reduction != "none":
                raise ValueError("If batch_dice is True, reduction cannot be None")
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        # perform reduction
        if self.reduction == "mean":
            dc = dc.mean()
        elif self.reduction == "mean_channels":
            dc = dc.mean(1)
        elif self.reduction == "sum":
            dc = dc.sum()
        return -dc


class DiceBCEloss(DC_and_BCE_loss):
    def __init__(
        self,
        bce_kwargs,
        soft_dice_kwargs,
        weight_ce=1,
        weight_dice=1,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLossWithReduction,
        reduction="none",
    ):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs["reduction"] = "none"
        if "reduction" in bce_kwargs:
            raise NotImplementedError(
                "reduction in bce_kwargs is not supported. Use reduction in this class instead"
            )
        if "reduction" in soft_dice_kwargs:
            raise NotImplementedError(
                "reduction in soft_dice_kwargs is not supported. Use reduction in this class instead"
            )
        if reduction.lower() not in ["none", "mean", "sum"]:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'. Got %s" % reduction)
        bce_kwargs["reduction"] = reduction
        if reduction == "none":
            soft_dice_kwargs["reduction"] = "mean_channels"
        else:
            soft_dice_kwargs["reduction"] = reduction
        self.reduction = reduction.lower()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(
                mask.sum(), min=1e-8
            )
        else:
            ce_loss = self.ce(net_output, target_regions)
        if self.reduction == "none":
            # reduce all but batch dimensions (Dice loss has only this dimension)
            ce_loss = torch.mean(ce_loss, dim=tuple(range(1, len(ce_loss.shape))), keepdim=True)
            dc_loss = dc_loss.view(*ce_loss.shape)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
