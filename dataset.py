import torch
import tqdm
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode

# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
orgId2trainId = torch.full((256,), -1, dtype=torch.long)
orgId2trainId[7] = 0  # road
orgId2trainId[8] = 1  # sidewalk
orgId2trainId[11] = 2  # building
orgId2trainId[12] = 3  # wall
orgId2trainId[13] = 4  # fence
orgId2trainId[17] = 5  # pole
orgId2trainId[19] = 6  # traffic light
orgId2trainId[20] = 7  # traffic sign
orgId2trainId[21] = 8  # vegetation
orgId2trainId[22] = 9  # terrain
orgId2trainId[23] = 10  # sky
orgId2trainId[24] = 11  # person
orgId2trainId[25] = 12  # rider
orgId2trainId[26] = 13  # car
orgId2trainId[27] = 14  # truck
orgId2trainId[28] = 15  # bus
orgId2trainId[31] = 16  # train
orgId2trainId[32] = 17  # motorcycle
orgId2trainId[33] = 18  # bicycle


def originalId2trainId(img):
    img = img.squeeze(0).long().clamp(0, 255)
    return orgId2trainId[
        img
    ]  # 6:10 -> 5:30 with smarter indexing (next time cProfile before!)


img_transform = transforms.Compose(
    [
        transforms.Resize((256, 512), InterpolationMode.NEAREST),
        transforms.ToTensor(),  # Also scales pixel values [0, 255] -> [0.0, 1.0]
    ]
)
tar_transform = transforms.Compose(
    [
        transforms.Resize((256, 512), InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(originalId2trainId),
    ]
)


def dataset():
    """Return a tuple with (train_dataset, validation_dataset)."""

    train_ds = Cityscapes(
        root="data",
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=tar_transform,
    )
    val_ds = Cityscapes(
        root="data",
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=tar_transform,
    )

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = dataset()
    print(len(train_ds))
    print(len(val_ds))

    r = tqdm.trange(100)
    for i in r:
        sample_img, sample_mask = train_ds[i]
        assert sample_img.shape == (3, 256, 512), (
            f"Wrong image shape: {sample_img.shape}"
        )
        assert sample_mask.shape == (256, 512), f"Wrong mask shape: {sample_mask.shape}"
        assert sample_mask.dtype == torch.long, (
            f"Mask should be long, got {sample_mask.dtype}"
        )
        assert sample_mask.min() >= -1, (
            f"Mask min should be -1, got {sample_mask.min()}"
        )
        assert sample_mask.max() <= 18, (
            f"Mask max should be 18, got {sample_mask.max()}"
        )

        groups = set()
        for row in sample_mask.squeeze(0):
            for p in row:
                groups.add(p.item())

        r.desc = f"{' '.join([f'{x}' for x in groups]):42}"
