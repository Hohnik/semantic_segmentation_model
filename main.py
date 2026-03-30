import os
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
epochs = 100
resume_checkpoint = "checkpoints/last.ckpt"  # Set to a path or None


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


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )

    encoder = MobileNetV2Encoder(width_mult=1.0, depth_mult=1.0)
    decoder = MobileNetV2Decoder(width_mult=1.0)

    model = EfficientUNetSegmentation(encoder, decoder).to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    miou = MeanIoU(19, include_background=False).to(device)
    accuracy = MulticlassAccuracy(num_classes=19, ignore_index=-1, average="micro").to(
        device
    )
    writer = SummaryWriter(log_dir="tb_logs/EffUNetSemSeg")

    writer.add_text("Learning Rate", str(lr))
    writer.add_text("Learning Rate", str(lr))
    writer.add_text("Weight Decay", weight_decay)
    writer.add_text("Batch Size", batch_size)
    writer.add_text("Epochs", epochs)
    writer.add_text("CPU/GPU", device.type)

    # MACs 570 Mio
    example_input = torch.randn(1, 3, 256, 512).to(device)
    macs = profile_macs(model.eval(), example_input)
    writer.add_text("MACs", macs)

    os.makedirs("checkpoints", exist_ok=True)

    start_epoch = 0
    best_val_miou = 0.0

    # Checkpoint continuing
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("val_miou", 0.0)
        print(
            f"Resuming from epoch {start_epoch} with best val mIoU {best_val_miou:.4f}"
        )

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        miou.reset()
        accuracy.reset()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            miou.update(preds, labels)
            accuracy.update(preds, labels)

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

            writer.add_scalar(
                "lr",
                optimizer.param_groups[0]["lr"],
                epoch * len(train_loader) + batch_idx,
            )

        avg_train_loss = train_loss / len(train_loader)
        train_miou = miou.compute()
        train_acc = accuracy.compute()
        writer.add_scalar("train_loss", avg_train_loss, epoch)
        writer.add_scalar("train_miou", train_miou, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        miou.reset()
        accuracy.reset()
        val_images, val_preds, val_labels = None, None, None

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, ignore_index=-1
                )

                preds = torch.argmax(logits, dim=1)
                miou.update(preds, labels)
                accuracy.update(preds, labels)

                val_loss += loss.item()
                val_images, val_preds, val_labels = images, preds, labels
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_miou = miou.compute()
        val_acc = accuracy.compute()

        writer.add_scalar("val_loss", avg_val_loss, epoch)
        writer.add_scalar("val_miou", val_miou, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)

        log_images(writer, val_images, val_preds, val_labels, epoch)

        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val mIoU {val_miou:.4f} | Val Acc {val_acc:.4f}"
        )

        # Checkpointing
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_miou": val_miou.item(),
        }
        torch.save(checkpoint, "checkpoints/last.ckpt")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(
                checkpoint, f"checkpoints/EffUNetSeg-{epoch:02d}-{val_miou:.2f}.ckpt"
            )


if __name__ == "__main__":
    main()
