import os
from pathlib import Path
import torch
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import dataset
from decoder import MobileNetV2Decoder
from encoder import MobileNetV2Encoder
from torchprofile import profile_macs
from torchvision.utils import make_grid

torch.serialization.add_safe_globals([MobileNetV2Encoder, MobileNetV2Decoder])

lr = 0.003
weight_decay = 0.0003
batch_size = 6
epochs = 10
checkpoints = Path("checkpoints")
checkpoints.mkdir(exist_ok=True)


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )

    train_data, val_data = dataset()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=(os.cpu_count() or 6) - 2,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=(os.cpu_count() or 6) - 2,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
    )

    models = []
    model_writers = []
    model_optimizers = []
    model_schedulers = []
    model_mious = []
    model_accuracies = []

    for width in [0.2, 1.0, 2.0]:
        for depth in [0.2, 1.0, 2.0]:
            writer = SummaryWriter(log_dir=f"tb_logs/EffUNetSemSeg-w{width}-d{depth}")

            writer.add_text("hparams/epochs", str(epochs))
            writer.add_text("hparams/device", device.type)
            writer.add_text("hparams/batch_size", str(batch_size))
            writer.add_text("hparams/learning_rate", str(lr))
            writer.add_text("hparams/weight_decay", str(weight_decay))
            writer.add_text("hparams/width_mult", str(width))
            writer.add_text("hparams/depth_mult", str(depth))

            encoder = MobileNetV2Encoder(width_mult=width, depth_mult=depth)
            decoder = MobileNetV2Decoder(width_mult=width)
            model = EfficientUNetSegmentation(encoder, decoder).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
            miou = MeanIoU(19, include_background=False).to(device)
            accuracy = MulticlassAccuracy(
                num_classes=19, ignore_index=-1, average="micro"
            ).to(device)

            # MACs 570 Mio
            example_input = torch.randn(1, 3, 256, 512).to(device)
            macs = profile_macs(model.eval(), example_input)
            writer.add_text("MACs", str(macs))

            # Data
            models.append(model)
            model_writers.append(writer)
            model_optimizers.append(optimizer)
            model_schedulers.append(scheduler)
            model_mious.append(miou)
            model_accuracies.append(accuracy)

    start_epoch = 0
    best_val_mious = [0.0 for _ in range(len(models))]

    # Training loop
    for epoch in range(start_epoch, epochs):
        train_losses = [0.0 for _ in range(len(models))]
        avg_train_losses = [0.0 for _ in range(len(models))]
        for miou, accuracy in zip(model_mious, model_accuracies):
            miou.reset()
            accuracy.reset()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            for i, (model, writer, optimizer, scheduler, miou, accuracy) in enumerate(
                zip(
                    models,
                    model_writers,
                    model_optimizers,
                    model_schedulers,
                    model_mious,
                    model_accuracies,
                )
            ):
                model.train()
                optimizer.zero_grad()
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, ignore_index=-1
                )
                loss.backward()
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                miou.update(preds, labels)
                accuracy.update(preds, labels)

                train_losses[i] += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})

                writer.add_scalar(
                    "lr",
                    optimizer.param_groups[0]["lr"],
                    epoch * len(train_loader) + batch_idx,
                )

        for i, (
            miou,
            accuracy,
            writer,
        ) in enumerate(zip(model_mious, model_accuracies, model_writers)):
            avg_train_losses[i] = train_losses[i] / len(train_loader)
            train_miou = miou.compute()
            train_acc = accuracy.compute()
            writer.add_scalar("train_loss", avg_train_losses[i], epoch)
            writer.add_scalar("train_miou", train_miou, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)

        # Validation loop
        val_losses = [0.0 for _ in range(len(models))]
        avg_val_losses = [0.0 for _ in range(len(models))]
        for miou, accuracy in zip(model_mious, model_accuracies):
            miou.reset()
            accuracy.reset()
        val_images, val_preds, val_labels = [], [], []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                for i, (model, miou, accuracy) in enumerate(
                    zip(
                        models,
                        model_mious,
                        model_accuracies,
                    )
                ):
                    model.eval()

                    logits = model(images)
                    loss = torch.nn.functional.cross_entropy(
                        logits, labels, ignore_index=-1
                    )

                    preds = torch.argmax(logits, dim=1)
                    miou.update(preds, labels)
                    accuracy.update(preds, labels)

                    val_losses[i] += loss.item()
                    val_images.append(images)
                    val_preds.append(preds)
                    val_labels.append(labels)

        for i, (model, optimizer, scheduler, miou, accuracy, writer) in enumerate(
            zip(
                models,
                model_optimizers,
                model_schedulers,
                model_mious,
                model_accuracies,
                model_writers,
            )
        ):
            avg_val_losses[i] = val_losses[i] / len(val_loader)
            val_miou = miou.compute()
            val_acc = accuracy.compute()

            writer.add_scalar("val_loss", avg_val_losses[i], epoch)
            writer.add_scalar("val_miou", val_miou, epoch)
            writer.add_scalar("val_acc", val_acc, epoch)

            log_images(writer, val_images[i], val_preds[i], val_labels[i], epoch)

            print(
                f"Epoch {epoch}: Train Loss {avg_train_losses[i]:.4f} | Val Loss {avg_val_losses[i]:.4f} | Val mIoU {val_miou:.4f} | Val Acc {val_acc:.4f}"
            )

            scheduler.step(avg_val_losses[i])

            # Checkpointing
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_miou": val_miou.item(),
            }

            if val_miou > best_val_mious[i]:
                best_val_mious[i] = val_miou
                torch.save(
                    checkpoint,
                    f"checkpoints/EffUNetSeg-{epoch:02d}-{val_miou:.2f}.ckpt",
                )


class EfficientUNetSegmentation(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        skips = self.encoder(x)
        logits = self.decoder(skips)
        return logits


def log_images(writer, images, preds, labels, epoch):
    img_grid = make_grid(images[:4], normalize=True)
    pred_grid = make_grid(preds[:4].unsqueeze(1).float() / 19)
    label_grid = make_grid(labels[:4].unsqueeze(1).float() / 19)

    writer.add_image("input", img_grid, epoch)
    writer.add_image("preds", pred_grid, epoch)
    writer.add_image("labels", label_grid, epoch)


if __name__ == "__main__":
    main()
