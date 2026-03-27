import os
import torch
from torchmetrics.segmentation import MeanIoU
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    miou_metric = MeanIoU(19).to(device)
    writer = SummaryWriter(log_dir="tb_logs/EffUNetSemSeg")

    # MACs 570 Mio
    example_input = torch.randn(1, 3, 256, 512).to(device)
    macs = profile_macs(model.eval(), example_input)
    writer.add_scalar("MACs", macs)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_miou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        miou_metric.reset()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            miou_metric.update(preds, labels)

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

            writer.add_scalar(
                "lr",
                optimizer.param_groups[0]["lr"],
                epoch * len(train_loader) + batch_idx,
            )

        avg_train_loss = train_loss / len(train_loader)
        train_miou = miou_metric.compute()
        writer.add_scalar("train_loss", avg_train_loss, epoch)
        writer.add_scalar("train_miou", train_miou, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        miou_metric.reset()
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
                miou_metric.update(preds, labels)

                val_loss += loss.item()
                val_images, val_preds, val_labels = images, preds, labels
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_miou = miou_metric.compute()

        writer.add_scalar("val_loss", avg_val_loss, epoch)
        writer.add_scalar("val_miou", val_miou, epoch)

        log_images(writer, val_images, val_preds, val_labels, epoch)

        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val mIoU {val_miou:.4f}"
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
