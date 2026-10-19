from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import CamVidDataset, CLASSES, VOID, MEAN, STD
from model import build_model, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "camvid"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "results"
FIG = OUT / "figures"
N = NUM_CLASSES

# Navigation grouping: 0 = Drivable, 1 = Obstacle, 2 = Other (non-drivable, non-obstacle)
GROUP = np.full(N, 1)
GROUP[3] = 0            # Road
GROUP[[0, 4]] = 2       # Sky, Pavement
GROUP_NAMES = ["Drivable", "Obstacle", "Other"]
NAV_COLORS = np.array([[0, 200, 0], [220, 30, 30], [140, 140, 140]], dtype=np.uint8)

# Standard CamVid colors (+ black for Void)
PALETTE = np.array([
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]], dtype=np.uint8)

H0 = 360  # original CamVid height (the dataset pads to 384)


def metrics(conf):
    conf = conf.astype(np.float64)
    denom = conf.sum(0) + conf.sum(1)
    iou = np.diag(conf) / np.maximum(denom - np.diag(conf), 1)
    dice = 2 * np.diag(conf) / np.maximum(denom, 1)
    acc = np.diag(conf).sum() / conf.sum()
    return iou, dice, acc


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(CKPT / "best.pt", map_location=dev))
    model.eval()

    ds = CamVidDataset(DATA, "test")
    dl = DataLoader(ds, 4, shuffle=False, num_workers=0)

    conf = torch.zeros(N * N, dtype=torch.long, device=dev)
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            pred = model(x).argmax(1)
            keep = y != VOID
            conf += torch.bincount(y[keep] * N + pred[keep], minlength=N * N)
    conf = conf.view(N, N).cpu().numpy()  # rows = ground truth, cols = prediction

    lines = [f"TEST SET ({len(ds)} images) - best.pt", ""]
    iou, dice, acc = metrics(conf)
    lines += ["11-class results:", f"  {'class':<11} {'IoU':>7} {'Dice':>7}"]
    lines += [f"  {n:<11} {i:7.4f} {d:7.4f}" for n, i, d in zip(CLASSES, iou, dice)]
    lines += [f"  {'MEAN':<11} {iou.mean():7.4f} {dice.mean():7.4f}",
              f"  pixel accuracy: {acc:.4f}", ""]

    conf3 = np.zeros((3, 3))
    np.add.at(conf3, (GROUP[:, None], GROUP[None, :]), conf)
    iou3, dice3, acc3 = metrics(conf3)
    lines += ["3-class navigation results (Drivable / Obstacle / Other):",
              f"  {'group':<11} {'IoU':>7} {'Dice':>7}"]
    lines += [f"  {n:<11} {i:7.4f} {d:7.4f}" for n, i, d in zip(GROUP_NAMES, iou3, dice3)]
    lines += [f"  {'MEAN':<11} {iou3.mean():7.4f} {dice3.mean():7.4f}",
              f"  pixel accuracy: {acc3:.4f}", ""]

    obs_as_road = conf3[1, 0] / conf3[1].sum()
    road_as_obs = conf3[0, 1] / conf3[0].sum()
    lines += ["Navigation safety:",
              f"  obstacle pixels predicted as drivable: {100 * obs_as_road:.2f}%",
              f"  drivable pixels predicted as obstacle: {100 * road_as_obs:.2f}%",
              "  dynamic obstacles predicted as Road:"]
    for c in (8, 9, 10):
        lines.append(f"    {CLASSES[c]:<11} {100 * conf[c, 3] / conf[c].sum():.2f}%")

    text = "\n".join(lines)
    print(text)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "test_metrics.txt").write_text(text, encoding="utf-8")

    # Qualitative figure
    idxs = np.linspace(0, len(ds) - 1, 6, dtype=int)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(14, 2.6 * len(idxs)))
    titles = ["Image", "Ground truth", "Prediction", "Navigation map"]
    for r, i in enumerate(idxs):
        x, y = ds[i]
        with torch.no_grad():
            p = model(x[None].to(dev)).argmax(1)[0].cpu().numpy()
        img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
        gt = y.numpy()
        nav = NAV_COLORS[GROUP[p]] / 255.0
        panels = [img, PALETTE[gt], PALETTE[p], 0.55 * img + 0.45 * nav]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel[:H0])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])
    plt.tight_layout()
    path = FIG / "test_predictions.png"
    plt.savefig(path, dpi=130)
    print("\nSaved", path, "and", OUT / "test_metrics.txt")


if __name__ == "__main__":
    main()