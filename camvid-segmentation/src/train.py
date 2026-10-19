import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamVidDataset, CLASSES, VOID
from model import build_model, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "camvid"
CKPT = ROOT / "checkpoints"


def class_weights(split="train"):
    """Median-frequency weights, softened with sqrt so rare classes don't dominate."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p in sorted((DATA / f"{split}annot").glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        counts += np.bincount(m.ravel(), minlength=VOID + 1)[:NUM_CLASSES]
    freq = counts / counts.sum()
    w = np.sqrt(np.median(freq) / freq)
    return torch.tensor(w, dtype=torch.float32)


def safe_save(obj, path):
    """Write to a temp file, then swap in, so a kill mid-save can't corrupt the checkpoint."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)  # fp32 on purpose
        keep = y != VOID
        idx = y[keep] * NUM_CLASSES + pred[keep]
        conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    conf = conf.double()
    denom = conf.sum(0) + conf.sum(1)
    iou = conf.diag() / (denom - conf.diag()).clamp(min=1)
    dice = 2 * conf.diag() / denom.clamp(min=1)
    acc = conf.diag().sum() / conf.sum()
    return iou.cpu().numpy(), dice.cpu().numpy(), acc.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (off by default)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    CKPT.mkdir(exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0), "| AMP:", args.amp)

    train_ds = CamVidDataset(DATA, "train")
    val_ds = CamVidDataset(DATA, "val")
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    model = build_model().to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights().to(device), ignore_index=VOID)
    dice_loss = smp.losses.DiceLoss("multiclass", ignore_index=VOID)
    loss_fn = lambda p, y: ce(p, y) + dice_loss(p, y)

    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": 1e-4}, {"params": rest, "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start, best_miou, best_iou, best_dice = 1, 0.0, None, None
    if args.resume and (CKPT / "last.pt").exists():
        ck = torch.load(CKPT / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best_miou = ck["epoch"] + 1, ck["best_miou"]
        print(f"Resumed from epoch {ck['epoch']} (best mIoU so far {best_miou:.4f})")

    for epoch in range(start, args.epochs + 1):
        model.train()
        total, n_ok, n_bad = 0.0, 0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            if not torch.isfinite(loss):  # skip bad batches instead of ruining the weights
                n_bad += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n_ok += 1
        sched.step()

        iou, dice_s, acc = evaluate(model, val_dl, device)
        miou, mdice = float(iou.mean()), float(dice_s.mean())
        print(f"epoch {epoch:02d} | loss {total / max(n_ok, 1):.4f} | "
              f"val mIoU {miou:.4f} | val Dice {mdice:.4f} | pixel acc {acc:.4f}"
              + (f" | skipped {n_bad} non-finite batches" if n_bad else ""))

        if miou > best_miou:
            best_miou, best_iou, best_dice = miou, iou, dice_s
            safe_save(model.state_dict(), CKPT / "best.pt")

        safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                   "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": epoch, "best_miou": best_miou}, CKPT / "last.pt")

    print(f"\nBest val mIoU: {best_miou:.4f}")
    if best_iou is not None:
        print(f"  {'class':<11} {'IoU':>7} {'Dice':>7}")
        for name, i, d in zip(CLASSES, best_iou, best_dice):
            print(f"  {name:<11} {i:7.4f} {d:7.4f}")


if __name__ == "__main__":
    main()