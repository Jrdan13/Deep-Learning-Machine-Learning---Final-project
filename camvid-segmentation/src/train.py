import argparse
import csv
import json
import os
import random
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
RUNS = ROOT / "results" / "runs"
FIELDS = ["epoch", "loss", "val_miou", "val_dice", "pixel_acc"]


def class_weights(split="train"):
    """Median-frequency weights, softened with sqrt so rare classes don't dominate."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p in sorted((DATA / f"{split}annot").glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        counts += np.bincount(m.ravel(), minlength=VOID + 1)[:NUM_CLASSES]
    freq = counts / counts.sum()
    return torch.tensor(np.sqrt(np.median(freq) / freq), dtype=torch.float32)


def safe_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
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
    ap.add_argument("--name", required=True, help="run name, e.g. base / aug / loss / both")
    ap.add_argument("--aug", choices=["none", "mild", "strong"], default="none")
    ap.add_argument("--loss", choices=["ce", "ce_dice"], default="ce")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["unet", "fcn", "deeplabv3plus"], default="unet")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ck_dir, log_dir = CKPT / args.name, RUNS / args.name
    log_path = log_dir / "log.csv"
    if log_path.exists() and not args.resume:
        raise SystemExit(f"Run '{args.name}' already exists. Use another --name, or --resume.")
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Run '{args.name}' | {torch.cuda.get_device_name(0)} | aug={args.aug} | "
          f"loss={args.loss} | class weights={args.class_weights} | seed={args.seed}")

    train_dl = DataLoader(CamVidDataset(DATA, "train", args.aug), args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(CamVidDataset(DATA, "val"), args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch).to(device)
    w = class_weights().to(device) if args.class_weights else None
    ce = nn.CrossEntropyLoss(weight=w, ignore_index=VOID)
    dice_loss = smp.losses.DiceLoss("multiclass", ignore_index=VOID)
    loss_fn = (lambda p, y: ce(p, y) + dice_loss(p, y)) if args.loss == "ce_dice" else ce

    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": 1e-4}, {"params": rest, "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start, best_miou, best_epoch, best_iou, best_dice = 1, 0.0, 0, None, None
    if args.resume and (ck_dir / "last.pt").exists():
        ck = torch.load(ck_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_miou, best_epoch = ck["epoch"] + 1, ck["best_miou"], ck["best_epoch"]
        print(f"Resumed from epoch {ck['epoch']} (best val mIoU so far {best_miou:.4f})")

    for epoch in range(start, args.epochs + 1):
        model.train()
        total, n_ok, n_bad = 0.0, 0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):  # skip bad batches instead of ruining the weights
                n_bad += 1
                continue
            loss.backward()
            opt.step()
            total += loss.item()
            n_ok += 1
        sched.step()

        iou, dice_s, acc = evaluate(model, val_dl, device)
        miou, mdice = float(iou.mean()), float(dice_s.mean())
        avg_loss = total / max(n_ok, 1)
        print(f"epoch {epoch:02d} | loss {avg_loss:.4f} | val mIoU {miou:.4f} | "
              f"val Dice {mdice:.4f} | pixel acc {acc:.4f}"
              + (f" | skipped {n_bad} non-finite batches" if n_bad else ""))

        new_file = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            wr = csv.writer(f)
            if new_file:
                wr.writerow(FIELDS)
            wr.writerow([epoch, f"{avg_loss:.4f}", f"{miou:.4f}", f"{mdice:.4f}", f"{acc:.4f}"])

        if miou > best_miou:
            best_miou, best_epoch, best_iou, best_dice = miou, epoch, iou, dice_s
            safe_save(model.state_dict(), ck_dir / "best.pt")
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                   "sched": sched.state_dict(), "epoch": epoch,
                   "best_miou": best_miou, "best_epoch": best_epoch}, ck_dir / "last.pt")

    with open(log_path, newline="") as f:
        vals = [float(r["val_miou"]) for r in csv.DictReader(f)]
    lines = [f"Run {args.name} (aug={args.aug}, loss={args.loss}, "
             f"class weights={args.class_weights}, seed={args.seed})",
             f"best val mIoU {best_miou:.4f} at epoch {best_epoch}",
             f"mean val mIoU over the last 5 epochs {np.mean(vals[-5:]):.4f}"]
    if best_iou is not None:
        lines += [f"  {n:<11} IoU {i:.4f}  Dice {d:.4f}"
                  for n, i, d in zip(CLASSES, best_iou, best_dice)]
    text = "\n".join(lines)
    print("\n" + text)
    (log_dir / "summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()