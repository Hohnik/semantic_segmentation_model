from torchvision.datasets import Cityscapes

train_ds = Cityscapes(root="data", split="train", mode="fine", target_type="semantic")
val_ds = Cityscapes(root="data", split="val", mode="fine", target_type="semantic")
