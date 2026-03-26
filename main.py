from typing import Literal

import lightning as L
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


def main(trial: optuna.Trial):
    optimizer_name = trial.suggest_categorical("optimizer_name", ["AdamW", "SGD"])
    scheduler_name = trial.suggest_categorical(
        "scheduler_name", ["CosineAnnealing", "ReduceOnPlateau"]
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 16, log=True)

    if optimizer_name == "AdamW":
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        momentum = 0.0
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        weight_decay = 0.0
    else:
        raise ValueError("Define optimizer via optimizer_name")

    encoder = MobileNetV2Encoder()
    decoder = MobileNetV2Decoder()

    segmenter = EfficientUNetSegmentation(
        encoder,
        decoder,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        batch_size=batch_size,
    )

    checkpoints = ModelCheckpoint(
        "checkpoints",
        filename="EffUNetSeg-{epoch}-{val_iou:.2f}",
        monitor="val_iou",
        save_last=True,
        save_weights_only=False,
    )
    progress_bar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor()

    logger = TensorBoardLogger("tb_logs", name="EfficientUNetSegmentation")
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=10,
        accelerator="gpu"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu",
        logger=logger,
        callbacks=[checkpoints, progress_bar, lr_monitor],
    )
    trainer.fit(
        model=segmenter,
    )

    return trainer.callback_metrics.get("val_iou", 0.0).item()


class EfficientUNetSegmentation(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr,
        momentum,
        weight_decay,
        batch_size,
        optimizer_name: Literal["AdamW", "SGD"] | str,
        scheduler_name: Literal["CosineAnnealing", "ReduceOnPlateau"] | str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[encoder, decoder])
        self.train_data, self.val_data = dataset()
        self._logged_hyperparams = False
        self.encoder = encoder
        self.decoder = decoder
        self.miou = MeanIoU(
            19
        )  # https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams["batch_size"],
            num_workers=0,  # os.cpu_count() or 1,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.hparams["batch_size"],
            num_workers=0,  # os.cpu_count() or 1
            # persistent_workers=True,
        )

    def forward(self, x):
        skips = self.encoder(x)
        logits = self.decoder(skips)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = torch.nn.functional.cross_entropy(
            input=logits, target=labels, ignore_index=-1
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
        miou = self.miou(preds, labels)

        self.log("val_iou", miou, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        self._log_images(images, preds, labels)

        return loss

    def on_validation_epoch_end(self):
        # Log hyperparameters with metric after first validation
        if not self._logged_hyperparams and self.current_epoch == 0:
            self.logger.log_hyperparams(
                self.hparams,
                {"hp_metric": self.trainer.callback_metrics.get("val_iou", 0.0)},
            )
            self._logged_hyperparams = True

    def configure_optimizers(self):  # type: ignore
        match self.hparams["optimizer_name"].lower():
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["weight_decay"],
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=self.hparams["momentum"],
                )
            case _:
                raise ValueError("Define optimizer via optimizer_name")

        match self.hparams["scheduler_name"].lower():
            case "cosineannealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=20, eta_min=1e-5
                )
            case "reduceonplateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            case _:
                raise ValueError("Define LearingRateScheduler via scheduler_name")

        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "val_iou",
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
    tensorboard_callback = TensorBoardCallback("tb_logs/optuna", metric_name="val_iou")

    # Delete existing optuna logs to avoid conflicts
    import shutil

    shutil.rmtree("tb_logs/optuna", ignore_errors=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="EffUNetSegOptimization",
    )
    study.optimize(main, n_trials=20, callbacks=[tensorboard_callback])

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
