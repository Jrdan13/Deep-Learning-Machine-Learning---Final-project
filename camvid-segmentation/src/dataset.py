from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
VOID = 11  # ignored in loss and metrics
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OUT_H, OUT_W = 384, 480  # 360x480 padded so the network sizes divide by 32


def color_transforms(aug):
    if aug == "mild":
        t = [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5),
             A.HueSaturationValue(p=0.3)]
    elif aug == "strong":
        t = [A.HorizontalFlip(p=0.5),
             A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.7),
             A.RandomGamma(gamma_limit=(60, 160), p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30,
                                  val_shift_limit=30, p=0.4),
             A.GaussianBlur(blur_limit=(3, 5), p=0.2)]
    else:
        t = []
    return A.Compose(t + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def pad_to(img, mask, h=OUT_H, w=OUT_W):
    ph, pw = max(h - mask.shape[0], 0), max(w - mask.shape[1], 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=VOID)
    return img, mask


def random_scale_crop(img, mask, smin=0.75, smax=1.5):
    """Random zoom, then crop (or pad with Void) back to 384x480."""
    s = np.random.uniform(smin, smax)
    h, w = mask.shape
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(OUT_H - nh, 0), max(OUT_W - nw, 0)
    if ph or pw:
        t, l = np.random.randint(0, ph + 1), np.random.randint(0, pw + 1)
        img = cv2.copyMakeBorder(img, t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=VOID)
    h, w = mask.shape
    y0, x0 = np.random.randint(0, h - OUT_H + 1), np.random.randint(0, w - OUT_W + 1)
    return img[y0:y0 + OUT_H, x0:x0 + OUT_W], mask[y0:y0 + OUT_H, x0:x0 + OUT_W]


class CamVidDataset(Dataset):
    """Expects data/camvid/{train,trainannot,val,valannot,test,testannot}.
    aug (none / mild / strong) only applies to the train split."""

    def __init__(self, root, split="train", aug="mild"):
        root = Path(root)
        self.images = sorted((root / split).glob("*.png"))
        self.masks = sorted((root / f"{split}annot").glob("*.png"))
        assert len(self.images) == len(self.masks) > 0, f"Bad data for split '{split}'"
        self.aug = aug if split == "train" else "none"
        self.tf = color_transforms(self.aug)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[i]), cv2.IMREAD_UNCHANGED)
        if self.aug == "strong" and np.random.rand() < 0.8:
            img, mask = random_scale_crop(img, mask)
        else:
            img, mask = pad_to(img, mask)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


if __name__ == "__main__":
    # Visual check: image and mask must stay aligned under every augmentation level.
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    pal = np.random.default_rng(0).integers(0, 255, (VOID + 1, 3), dtype=np.uint8)
    pal[VOID] = 0
    rows = ["none", "mild", "strong", "strong"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 3 * len(rows)))
    for r, aug in enumerate(rows):
        x, y = CamVidDataset(root / "data" / "camvid", "train", aug)[5]
        print(aug, tuple(x.shape), tuple(y.shape))
        im = (x.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        axes[r, 0].imshow(im); axes[r, 0].set_title(f"{aug}: image")
        axes[r, 1].imshow(pal[y.numpy()]); axes[r, 1].set_title("mask")
        axes[r, 0].axis("off"); axes[r, 1].axis("off")
    out = root / "results" / "figures" / "aug_check.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print("Saved", out)