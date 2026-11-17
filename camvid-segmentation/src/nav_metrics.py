from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import CamVidDataset, CLASSES, VOID, MEAN, STD
from model import build_model
from evaluate import GROUP, PALETTE, H0, DATA, CKPT, OUT, FIG

ROAD, PAVE, CAR = 3, 4, 8
H, W = 384, 480


def roi_mask():
    """Trapezoid in front of the vehicle (fractions of the 360x480 image). Tweak if needed."""
    pts = np.array([[int(0.05 * W), H0], [int(0.95 * W), H0],
                    [int(0.62 * W), int(0.50 * H0)], [int(0.38 * W), int(0.50 * H0)]], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool), pts


def report(name, c):
    c = c.astype(float)
    s = lambda v: max(v, 1.0)
    return [name,
            f"  obstacle called drivable:            {100 * c[1, 0] / s(c[1].sum()):.2f}%",
            f"  other (pavement/sky) called drivable: {100 * c[2, 0] / s(c[2].sum()):.2f}%",
            f"  drivable called not-drivable:         {100 * (1 - c[0, 0] / s(c[0].sum())):.2f}%",
            f"  pixels called drivable that are not:  {100 * (c[1, 0] + c[2, 0]) / s(c[:, 0].sum()):.2f}%",
            f"  drivable IoU:                         {c[0, 0] / s(c[0].sum() + c[:, 0].sum() - c[0, 0]):.4f}"]


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(CKPT / "best.pt", map_location=dev))
    model.eval()

    ds = CamVidDataset(DATA, "test")
    dl = DataLoader(ds, 4, shuffle=False, num_workers=0)
    group = torch.tensor(GROUP, device=dev)
    roi, pts = roi_mask()
    roi_t = torch.from_numpy(roi).to(dev)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    full = torch.zeros(9, dtype=torch.long, device=dev)
    inroi = torch.zeros(9, dtype=torch.long, device=dev)
    car_err, car_tot, bright = [], [], []
    pf = [0, 0, 0, 0]  # pavement total/error: full image, ROI
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        keep = y != VOID
        idx = group[y.clamp(max=VOID - 1)] * 3 + group[pred]
        full += torch.bincount(idx[keep], minlength=9)
        inroi += torch.bincount(idx[keep & roi_t], minlength=9)
        car_err += ((y == CAR) & (pred == ROAD)).sum((1, 2)).tolist()
        car_tot += (y == CAR).sum((1, 2)).tolist()
        bright += (x * std + mean)[:, :, :H0].mean((1, 2, 3)).tolist()
        pv, pr = y == PAVE, pred == ROAD
        pf[0] += pv.sum().item();           pf[1] += (pv & pr).sum().item()
        pf[2] += (pv & roi_t).sum().item(); pf[3] += (pv & pr & roi_t).sum().item()

    car_err, car_tot, bright = np.array(car_err), np.array(car_tot), np.array(bright)
    lines = ["NAVIGATION METRICS (test set, best.pt)", ""]
    lines += report("Whole image:", full.cpu().numpy().reshape(3, 3)) + [""]
    lines += report("Inside driving ROI:", inroi.cpu().numpy().reshape(3, 3)) + [""]
    lines += [f"Pavement called Road: whole image {100 * pf[1] / max(pf[0], 1):.2f}% | "
              f"ROI {100 * pf[3] / max(pf[2], 1):.2f}%", ""]
    top = np.argsort(car_err)[::-1][:6]
    lines += [f"Car called Road (whole image): {100 * car_err.sum() / max(car_tot.sum(), 1):.2f}%",
              f"Mean brightness (0-1): all test images {bright.mean():.3f} | worst 6 images {bright[top].mean():.3f}",
              "Worst images (index: car->road pixels / car pixels, brightness):"]
    lines += [f"  #{i}: {car_err[i]} / {car_tot[i]}, {bright[i]:.3f}" for i in top]
    text = "\n".join(lines)
    print(text)
    (OUT / "nav_metrics.txt").write_text(text, encoding="utf-8")

    # Figure 1: worst Car->Road images (red = car pixels predicted as road)
    fig, axes = plt.subplots(len(top), 4, figsize=(14, 2.6 * len(top)))
    for c, t in enumerate(["Image", "Ground truth", "Prediction", "Car called road (red)"]):
        axes[0, c].set_title(t)
    for r, i in enumerate(top):
        x, y = ds[int(i)]
        p = model(x[None].to(dev)).argmax(1)[0].cpu().numpy()
        gt = y.numpy()
        img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
        over = img.copy()
        over[(gt == CAR) & (p == ROAD)] = [1, 0, 0]
        for c, panel in enumerate([img, PALETTE[gt], PALETTE[p], over]):
            axes[r, c].imshow(panel[:H0]); axes[r, c].axis("off")
        axes[r, 0].text(6, 28, f"#{i}", color="white", fontsize=10, bbox=dict(fc="black", alpha=0.6))
    plt.tight_layout(); plt.savefig(FIG / "car_as_road.png", dpi=120); plt.close()

    # Figure 2: where the ROI sits
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for a, i in zip(axs, [0, len(ds) // 2]):
        x, _ = ds[i]
        img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
        a.imshow(img[:H0]); a.add_patch(plt.Polygon(pts, fill=False, ec="yellow", lw=2)); a.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "roi_check.png", dpi=120); plt.close()
    print("\nSaved nav_metrics.txt, car_as_road.png, roi_check.png in results/")


if __name__ == "__main__":
    main()