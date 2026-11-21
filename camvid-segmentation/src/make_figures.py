"""Report figures. Reads results/runs/<name>/log.csv, results/runs/<name>/test_per_class.txt,
results/ablation.csv and results/efficiency.csv. Writes PNG (300 dpi) and PDF to
results/figures/paper/. Runs that do not exist yet are skipped, so it is safe to run at any time."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
RECIPES = {  # 2x2 ablation, two seeds each
    "Baseline\n(no aug, CE)": ["base", "base_s1"],
    "+ Strong aug": ["aug", "aug_s1"],
    "+ Dice": ["loss", "loss_s1"],
    "+ Aug + Dice": ["both", "both_s1"],
}
MODELS = {  # same recipe (strong aug, CE + Dice), two seeds each
    "U-Net": ["both", "both_s1"],
    "FCN-8s": ["fcn_both", "fcn_both_s1"],
    "DeepLabV3+": ["dlv3p_both", "dlv3p_both_s1"],
}
ARCH_LABEL = {"unet": "U-Net", "fcn": "FCN-8s", "deeplabv3plus": "DeepLabV3+"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


# ---------- readers ----------
def read_log(name):
    p = RUNS / name / "log.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ["epoch", "loss", "val_miou", "val_dice", "pixel_acc"]}


def stack(names, key):
    logs = [l for l in (read_log(n) for n in names) if l is not None]
    if not logs:
        return None
    n = min(len(l[key]) for l in logs)
    return np.stack([l[key][:n] for l in logs]), logs[0]["epoch"][:n]


def read_table():
    p = ROOT / "results" / "ablation.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


TABLE = read_table()


def metric(names, key):
    return [float(TABLE[n][key]) for n in names if n in TABLE and TABLE[n].get(key, "") != ""]


def per_class(name):
    p = RUNS / name / "test_per_class.txt"
    if not p.exists():
        return None
    vals = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\w+)\s+IoU\s+([\d.]+)", line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    return np.array([vals[c] for c in CLASSES]) if all(c in vals for c in CLASSES) else None


# ---------- curves ----------
def draw_curves(ax, groups, key):
    tails = []
    for (label, names), c in zip(groups.items(), COLORS):
        s = stack(names, key)
        if s is None:
            continue
        arr, ep = s
        tails.append(arr[:, 19:].ravel())
        ax.plot(ep, arr.mean(0), color=c, lw=1.8, label=f"{label.replace(chr(10), ' ')} (n={len(arr)})")
        ax.fill_between(ep, arr.min(0), arr.max(0), color=c, alpha=0.15)
    return np.concatenate(tails) if tails else None


def fig_val_curves(groups, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, zoom in zip(axes, [False, True]):
        tail = draw_curves(ax, groups, "val_miou")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mIoU")
        if zoom and tail is not None:
            ax.set_xlim(20, 40)
            ax.set_ylim(tail.min() - 0.005, tail.max() + 0.005)
            ax.set_title("Zoom: epochs 20-40")
        else:
            ax.set_ylim(0.35, 0.80)
            ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, fname)


def fig_train_loss():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sets = [("Cross-entropy only", {"No aug": ["base", "base_s1"], "Strong aug": ["aug", "aug_s1"]}),
            ("Cross-entropy + Dice", {"No aug": ["loss", "loss_s1"], "Strong aug": ["both", "both_s1"]})]
    for ax, (t, g) in zip(axes, sets):
        draw_curves(ax, g, "loss")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.legend(fontsize=8)
    fig.suptitle("Training loss (values are not comparable across loss functions)", y=1.02)
    fig.tight_layout()
    save(fig, "fig_train_loss")


# ---------- bars ----------
def bars(ax, entries, key, ylabel, scale=1.0, zero=False, fmt="{:.3f}"):
    labels, means, pts = [], [], []
    for label, names in entries.items():
        v = metric(names, key)
        if v:
            labels.append(label)
            means.append(np.mean(v) * scale)
            pts.append(np.array(v) * scale)
    if not labels:
        ax.set_visible(False)
        return
    x = np.arange(len(labels))
    ax.bar(x, means, color=COLORS[:len(labels)], alpha=0.8, width=0.6)
    for i, p in enumerate(pts):
        ax.scatter(np.full(len(p), i), p, color="black", s=14, zorder=3)
        ax.text(i, max(p.max(), means[i]), fmt.format(means[i]), ha="center", va="bottom", fontsize=8)
    allv = np.concatenate(pts)
    span = max(allv.max() - allv.min(), 1e-6)
    ax.set_ylim(0 if zero else allv.min() - 1.0 * span, allv.max() + 0.35 * span if not zero else allv.max() * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)


def fig_bars(entries, fname, title):
    panels = [("test_miou", "Test mIoU", 1, False, "{:.3f}"),
              ("road_iou", "Road IoU (test)", 1, False, "{:.3f}"),
              ("roi_obstacle_to_drivable", "Obstacle called drivable, ROI (%)", 1, True, "{:.2f}"),
              ("car_to_road", "Car called road (%)", 1, True, "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (k, yl, sc, z, fm) in zip(axes, panels):
        bars(ax, entries, k, yl, sc, z, fm)
    fig.suptitle(title + " (bars: mean of 2 seeds, dots: individual seeds)", y=1.03)
    fig.tight_layout()
    save(fig, fname)


def fig_perclass(entries, fname, title):
    series = {}
    for label, names in entries.items():
        arrs = [a for a in (per_class(n) for n in names) if a is not None]
        if arrs:
            series[label.replace("\n", " ")] = np.mean(arrs, axis=0)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.8 / len(series)
    x = np.arange(len(CLASSES))
    for i, ((label, vals), c) in enumerate(zip(series.items(), COLORS)):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=label, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha="right")
    ax.set_ylabel("Test IoU (mean of 2 seeds)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(series))
    fig.tight_layout()
    save(fig, fname)


# ---------- speed vs accuracy ----------
def fig_speed():
    p = ROOT / "results" / "efficiency.csv"
    if not p.exists():
        return
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for r, c in zip(rows, COLORS):
        label = ARCH_LABEL.get(r["arch"], r["arch"])
        v = metric(MODELS.get(label, []), "test_miou")
        if not v:
            continue
        fps, miou = float(r["fps_batch1"]), float(np.mean(v))
        ax.scatter(fps, miou, s=float(r["params_M"]) * 18, color=c, alpha=0.8, edgecolor="k", zorder=3)
        ax.annotate(f"{label}\n{r['params_M']} M params, {r['GFLOPs']} GFLOPs", (fps, miou),
                    textcoords="offset points", xytext=(10, -22), fontsize=8)
    ax.margins(x=0.3, y=0.35)
    ax.axvline(30, color="gray", ls="--", lw=1)
    ax.text(30.5, ax.get_ylim()[0], " 30 FPS", color="gray", fontsize=8, va="bottom")
    ax.set_xlabel("Inference speed (FPS, batch 1, 384x480, fp32, T1200)")
    ax.set_ylabel("Test mIoU (mean of 2 seeds)")
    ax.set_title("Accuracy vs speed (bubble area = parameters)")
    fig.tight_layout()
    save(fig, "fig_speed_accuracy")


# ---------- flow chart ----------
def box(ax, cx, cy, title, body, color, w=4.4, h=1.9):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(cx, cy + h * 0.24, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(cx, cy - h * 0.14, body, ha="center", va="center", fontsize=8.5, linespacing=1.35)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#333333"))


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    xs, ys = [2.6, 8.0, 13.4], [5.9, 3.5, 1.1]
    data, model, train, ev = "#D6E8F7", "#FBE3C8", "#D9EFD9", "#E6DDF2"
    box(ax, xs[0], ys[0], "1  Dataset", "CamVid, 701 street images\n11 classes + Void (ignored)\ntrain 367 / val 101 / test 233", data)
    box(ax, xs[1], ys[0], "2  Preprocessing", "pad 360x480 to 384x480\nImageNet normalisation\nVoid label for padding", data)
    box(ax, xs[2], ys[0], "3  Augmentation (train only)", "flip, zoom 0.75-1.5x + crop\nbrightness, gamma, HSV, blur", data)
    box(ax, xs[2], ys[1], "4  Network", "ResNet34 encoder (ImageNet)\nU-Net | FCN-8s | DeepLabV3+ decoder\n11 class scores per pixel", model)
    box(ax, xs[1], ys[1], "5  Loss", "cross-entropy + Dice\nVoid pixels ignored", model)
    box(ax, xs[0], ys[1], "6  Optimisation", "AdamW, cosine schedule\n40 epochs, batch 4, fp32\nseeds 0 and 1", train)
    box(ax, xs[0], ys[2], "7  Model selection", "validation mIoU after every epoch\nkeep the best checkpoint", train)
    box(ax, xs[1], ys[2], "8  Test evaluation", "233 unseen images, used once\nmIoU, Dice, per-class IoU", ev)
    box(ax, xs[2], ys[2], "9  Navigation analysis", "Drivable / Obstacle / Other\nROI safety metrics, speed (FPS)", ev)
    g = 0.02
    for row in (0,):
        arrow(ax, (xs[0] + 2.2 + g, ys[row]), (xs[1] - 2.2 - g, ys[row]))
        arrow(ax, (xs[1] + 2.2 + g, ys[row]), (xs[2] - 2.2 - g, ys[row]))
    arrow(ax, (xs[2], ys[0] - 0.95 - g), (xs[2], ys[1] + 0.95 + g))
    arrow(ax, (xs[2] - 2.2 - g, ys[1]), (xs[1] + 2.2 + g, ys[1]))
    arrow(ax, (xs[1] - 2.2 - g, ys[1]), (xs[0] + 2.2 + g, ys[1]))
    arrow(ax, (xs[0], ys[1] - 0.95 - g), (xs[0], ys[2] + 0.95 + g))
    arrow(ax, (xs[0] + 2.2 + g, ys[2]), (xs[1] - 2.2 - g, ys[2]))
    arrow(ax, (xs[1] + 2.2 + g, ys[2]), (xs[2] - 2.2 - g, ys[2]))
    ax.set_title("Experimental pipeline", fontsize=13, fontweight="bold", pad=6)
    save(fig, "fig_flowchart_pipeline")


def main():
    fig_val_curves(RECIPES, "fig_val_curves_ablation", "Validation mIoU (mean and min-max of 2 seeds)")
    fig_val_curves(MODELS, "fig_val_curves_models", "Validation mIoU by architecture (mean and min-max)")
    fig_train_loss()
    fig_bars(RECIPES, "fig_bars_ablation", "Ablation, U-Net ResNet34")
    fig_bars(MODELS, "fig_bars_models", "Architectures, same recipe")
    fig_perclass(RECIPES, "fig_perclass_ablation", "Per-class test IoU, ablation")
    fig_perclass(MODELS, "fig_perclass_models", "Per-class test IoU, architectures")
    fig_speed()
    fig_flowchart()
    print("done ->", OUT)


if __name__ == "__main__":
    main()