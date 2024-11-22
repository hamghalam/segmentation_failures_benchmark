import monai
import pytorch_lightning as pl


class NetworkGraphViz(pl.Callback):
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int
    ) -> None:
        if trainer.global_step == 0:
            input_array = batch["data"]
            if isinstance(input_array, monai.data.MetaTensor):
                input_array = input_array.as_tensor()
            self.logger.log_graph(pl_module, input_array=input_array)
