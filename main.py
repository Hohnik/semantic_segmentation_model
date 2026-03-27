from typing import Literal
import os

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import optuna
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import TensorBoardCallback
from torchmetrics.segmentation import MeanIoU

from dataset import dataset
from decoder import MobileNetV2Decoder
from encoder import MobileNetV2Encoder

torch.serialization.add_safe_globals([MobileNetV2Encoder, MobileNetV2Decoder])

lr = 0.001
weight_decay = 0.0001
batch_size = 16


def main():
    encoder = MobileNetV2Encoder(width_mult=1.0, depth_mult=1.0)
    decoder = MobileNetV2Decoder(width_mult=1.0)

    semantic_segmentation = EfficientUNetSegmentation(
        encoder,
        decoder,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
    )

    checkpoints = ModelCheckpoint(
        "checkpoints",
        filename="EffUNetSeg-{epoch:02d}-{val_miou:.2f}",
        monitor="val_miou",
        save_last=True,
        save_weights_only=False,
    )
    progress_bar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger("tb_logs", name="EffUNetSemSeg")
    trainer = L.Trainer(
        fast_dev_run=10,
        accelerator="gpu"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu",
        logger=logger,
        callbacks=[checkpoints, progress_bar, lr_monitor],
        precision="16-mixed",
    )
    trainer.fit(
        model=semantic_segmentation,
    )


class EfficientUNetSegmentation(L.LightningModule):
    def __init__(self, encoder, decoder, lr, weight_decay, batch_size):
        super().__init__()
        self.saved_hparams = False
        self.hparams["lr"] = lr
        self.hparams["weight_decay"] = weight_decay
        self.hparams["batch_size"] = batch_size

        self.encoder = encoder
        self.decoder = decoder

        self.train_data, self.val_data = dataset()
        self.miou = MeanIoU(
            19
        )  # https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams["batch_size"],
            num_workers=(os.cpu_count() or 6) - 2,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.hparams["batch_size"],
            num_workers=(os.cpu_count() or 6) - 2,
            persistent_workers=True,
            pin_memory=True,
        )

    def forward(self, x):
        skips = self.encoder(x)
        logits = self.decoder(skips)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
        self.miou.update(preds, labels)
        self.curr_loss = loss.item()

        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.curr_loss)
        self.log("train_miou", self.miou.compute())

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
        self.curr_loss = loss.item()
        self.miou.update(preds, labels)
        self.curr_images = images
        self.curr_preds = preds
        self.curr_labels = labels

        return loss

    def on_validation_epoch_end(self):
        miou = self.miou.compute()
        self.log("val_miou", miou, prog_bar=True)
        self.log("val_loss", self.curr_loss, prog_bar=True)
        self._log_images(self.curr_images, self.curr_preds, self.curr_labels)

        if not self.saved_hparams:
            self.save_hyperparameters(ignore=[self.encoder, self.decoder])
            self.saved_hparams = True

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / 10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [warmup, cosine], optimizer
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        ]

    def _log_images(self, images, preds, labels):
        from torchvision.utils import (
            make_grid,
        )  # https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html -> Further improvements on visualization

        img_grid = make_grid(images[:4], normalize=True)
        pred_grid = make_grid(preds[:4].unsqueeze(1).float() / 19)
        label_grid = make_grid(labels[:4].unsqueeze(1).float() / 19)

        self.logger.experiment.add_image("input", img_grid, self.current_epoch)  # type: ignore
        self.logger.experiment.add_image("preds", pred_grid, self.current_epoch)  # type: ignore
        self.logger.experiment.add_image("labels", label_grid, self.current_epoch)  # type: ignore


if __name__ == "__main__":
    main()
