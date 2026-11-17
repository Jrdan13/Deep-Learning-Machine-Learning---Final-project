import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CamVidDataset, CLASSES, VOID
from model import build_model, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA, CKPT = ROOT / "data" / "camvid", ROOT / "checkpoints"
RUNS, TABLE = ROOT / "results" / "runs", ROOT / "results" / "ablation.csv"
N, H, W, H0 = NUM_CLASSES, 384, 480, 360
ROAD, PAVE, CAR = 3, 4, 8
GROUP = np.full(N, 1); GROUP[3] = 0; GROUP[[0, 4]] = 2  # 0 drivable, 1 obstacle, 2 other
COLS = ["name", "aug", "loss", "val_best_miou", "val_last5_miou", "test_miou", "test_dice",
        "road_iou", "roi_obstacle_to_drivable", "roi_pavement_to_road", "car_to_road"]


def roi_mask():  # same driving corridor as nav_metrics.py
    pts = np.array([[int(0.05 * W), H0], [int(0.95 * W), H0],
                    [int(0.62 * W), int(0.50 * H0)], [int(0.38 * W), int(0.50 * H0)]], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    name = ap.parse_args().name

    dev = torch.device("cuda")
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(CKPT / name / "best.pt", map_location=dev))
    model.eval()
    dl = DataLoader(CamVidDataset(DATA, "test"), 4, shuffle=False, num_workers=0)

    group = torch.tensor(GROUP, device=dev)
    roi = torch.from_numpy(roi_mask()).to(dev)
    conf = torch.zeros(N * N, dtype=torch.long, device=dev)
    c3 = torch.zeros(9, dtype=torch.long, device=dev)    # 3-group matrix inside the ROI
    pav = torch.zeros(2, dtype=torch.long, device=dev)   # ROI pavement pixels, called road
    car = torch.zeros(2, dtype=torch.long, device=dev)   # car pixels, called road
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        keep = y != VOID
        conf += torch.bincount(y[keep] * N + pred[keep], minlength=N * N)
        idx = group[y.clamp(max=VOID - 1)] * 3 + group[pred]
        c3 += torch.bincount(idx[keep & roi], minlength=9)
        pv = (y == PAVE) & roi
        pav += torch.stack([pv.sum(), (pv & (pred == ROAD)).sum()])
        cr = y == CAR
        car += torch.stack([cr.sum(), (cr & (pred == ROAD)).sum()])

    c = conf.view(N, N).cpu().numpy().astype(float)
    den = c.sum(0) + c.sum(1)
    iou = np.diag(c) / np.maximum(den - np.diag(c), 1)
    dice = 2 * np.diag(c) / np.maximum(den, 1)
    g = c3.view(3, 3).cpu().numpy().astype(float)
    pav, car = pav.cpu().numpy(), car.cpu().numpy()

    cfg_path = RUNS / name / "config.json"
    cfg = (json.loads(cfg_path.read_text()) if cfg_path.exists()
           else {"aug": "mild", "loss": "ce_dice", "class_weights": True})  # the original model
    loss_label = cfg["loss"] + ("+w" if cfg.get("class_weights") else "")
    vb = v5 = ""
    log_path = RUNS / name / "log.csv"
    if log_path.exists():
        with open(log_path) as f:
            v = [float(r["val_miou"]) for r in csv.DictReader(f)]
        vb, v5 = f"{max(v):.4f}", f"{np.mean(v[-5:]):.4f}"

    row = dict(zip(COLS, [
        name, cfg["aug"], loss_label, vb, v5, f"{iou.mean():.4f}", f"{dice.mean():.4f}",
        f"{iou[ROAD]:.4f}", f"{100 * g[1, 0] / g[1].sum():.2f}",
        f"{100 * pav[1] / max(pav[0], 1):.2f}", f"{100 * car[1] / max(car[0], 1):.2f}"]))

    rows = []
    if TABLE.exists():
        with open(TABLE, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != name]
    rows.append(row)
    with open(TABLE, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(rows)

    per_class = [f"  {n:<11} IoU {i:.4f}  Dice {d:.4f}" for n, i, d in zip(CLASSES, iou, dice)]
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name / "test_per_class.txt").write_text("\n".join(per_class), encoding="utf-8")
    print("| " + " | ".join(COLS) + " |")
    for r in rows:
        print("| " + " | ".join(r[k] for k in COLS) + " |")
    print(f"\nTest per-class ({name}):\n" + "\n".join(per_class))
    print("\nSaved", TABLE)


if __name__ == "__main__":
    main()