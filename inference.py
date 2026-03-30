from pathlib import Path
import torch
from encoder import MobileNetV2Encoder
from decoder import MobileNetV2Decoder
from main import EfficientUNetSegmentation
from dataset import img_transform
from torchvision.utils import draw_segmentation_masks, save_image
import numpy as np
from PIL import Image

COLORS = [
    (0, 0, 0),  # 0: black
    (255, 0, 0),  # 1: red
    (0, 128, 0),  # 2: green
    (0, 0, 255),  # 3: blue
    (255, 255, 0),  # 4: yellow
    (255, 0, 255),  # 5: magenta
    (0, 255, 255),  # 6: cyan
    (255, 255, 255),  # 7: white
    (255, 165, 0),  # 8: orange
    (128, 0, 128),  # 9: purple
    (255, 192, 203),  # 10: pink
    (165, 42, 42),  # 11: brown
    (128, 128, 128),  # 12: gray
    (128, 128, 0),  # 13: olive
    (0, 128, 128),  # 14: teal
    (0, 0, 128),  # 15: navy
    (128, 0, 0),  # 16: maroon
    (0, 255, 0),  # 17: lime
    (175, 238, 238),  # 18: aqua
    (192, 192, 192),  # 19: silver
]

# Load Images
inference_input = Path("inference/input/")
inference_output = Path("inference/output/")
image_paths = list(inference_input.glob("*"))
image_names = [image.name for image in image_paths]
images: list[torch.Tensor] = [
    img_transform(Image.open(image).convert("RGB")) for image in image_paths
]  # type: ignore

print(f"Images: {image_names}")

# Load Model
resume_checkpoint = Path("checkpoints/EffUNetSeg-27-0.98.ckpt")  # Set to a path or None
assert resume_checkpoint.exists(), (
    f"There there is no checkpoint saved at '{resume_checkpoint}'"
)
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
checkpoint = torch.load(resume_checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])


# Inference
batch = torch.stack(images).to(device)
with torch.no_grad():
    preds: torch.Tensor = model(batch).argmax(dim=1)

    for image, pred, name in zip(images, preds, image_names):
        # print(pred.unique())  # classes detected
        bool_masks = torch.nn.functional.one_hot(pred, 20).permute(2, 0, 1).bool()
        colored_image = draw_segmentation_masks(image, bool_masks, colors=COLORS)  # type:ignore

        arr = (colored_image * 255).byte().permute(1, 2, 0).numpy()  # [0,1] -> [0,255]
        Image.fromarray(arr).show()
        save_image(colored_image, inference_output / f"segmented_{name}")
