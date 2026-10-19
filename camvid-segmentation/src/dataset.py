from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
VOID = 11  # "Unlabelled" in the SegNet-Tutorial annotations, ignored in loss/metrics

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool):
    aug = []
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
        ]
    aug += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(aug)


class CamVidDataset(Dataset):
    """Expects data/camvid/{train,trainannot,val,valannot,test,testannot}."""

    def __init__(self, root, split="train"):
        root = Path(root)
        self.images = sorted((root / split).glob("*.png"))
        self.masks = sorted((root / f"{split}annot").glob("*.png"))
        assert len(self.images) == len(self.masks) > 0, f"Bad data for split '{split}'"
        self.tf = get_transforms(train=(split == "train"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[i]), cv2.IMREAD_UNCHANGED)

        # Pad 360x480 -> 384x480 (U-Net needs sizes divisible by 32).
        # Padded mask pixels are marked VOID so they are ignored.
        h, w = mask.shape
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=VOID)

        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


if __name__ == "__main__":
    # Sanity check: shapes, label values, class frequencies, and an overlay figure.
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    ds = CamVidDataset(root / "data" / "camvid", "train")
    x, y = ds[0]
    print("image:", tuple(x.shape), x.dtype, "| mask:", tuple(y.shape), y.dtype)

    counts = np.zeros(VOID + 1, dtype=np.int64)
    for _, m in ds:
        counts += np.bincount(m.numpy().ravel(), minlength=VOID + 1)
    print("\nPixel share per class (train):")
    for name, c in zip(CLASSES + ["Void"], counts):
        print(f"  {name:<11} {100 * c / counts.sum():5.2f}%")

    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, (VOID + 1, 3), dtype=np.uint8)
    palette[VOID] = 0
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    for row, idx in zip(axes, [0, 100, 200]):
        xi, yi = ds[idx]
        im = (xi.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        row[0].imshow(im); row[1].imshow(palette[yi.numpy()])
        row[0].axis("off"); row[1].axis("off")
    out = root / "results" / "figures" / "dataset_check.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print("\nSaved", out) 