from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class TrainingTargetMonitor(Callback):
    def __init__(
        self,
        output_dir: str,
        save_every_n_epochs: int = 1,
    ):
        """This callback should save the predicted and target quality values after each training epoch"""
        self.output_path = Path(output_dir)
        self.output_path.mkdir()
        self.save_every_n_epochs = save_every_n_epochs
        self._quality_true_buffer = []  # this is cleared after each epoch
        self._quality_pred_buffer = []  # this is cleared after each epoch

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int, unused: int = 0
    ) -> None:
        if trainer.current_epoch % self.save_every_n_epochs != 0:
            return
        self._quality_pred_buffer.append(outputs["quality_pred"])  # tensor of shape BM
        self._quality_true_buffer.append(outputs["quality_true"])  # tensor of shape BM

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.save_every_n_epochs != 0:
            return
        # log true quality values (just for sanity checking)
        all_quality_true = torch.cat(self._quality_true_buffer, dim=0).cpu().numpy()
        all_quality_pred = torch.cat(self._quality_pred_buffer, dim=0).cpu().numpy()
        if hasattr(trainer.datamodule, "metric_target_names"):
            quality_names = trainer.datamodule.metric_target_names
        np.savez(
            self.output_path / f"quality_targets_epoch={pl_module.current_epoch}.npz",
            quality_true=all_quality_true,
            quality_pred=all_quality_pred,
            names=quality_names,
        )
        self._quality_true_buffer = []
        self._quality_pred_buffer = []
