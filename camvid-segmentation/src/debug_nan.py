import torch, torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import CamVidDataset, VOID
from model import build_model
from train import class_weights, DATA

dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
x, y = next(iter(DataLoader(CamVidDataset(DATA, "train"), 4, shuffle=True)))
x, y = x.to(dev), y.to(dev)
print("x finite:", torch.isfinite(x).all().item(), "| labels:", y.unique().tolist())
w = class_weights(); print("weights:", [round(float(v), 2) for v in w])

m = build_model().to(dev)
ce = nn.CrossEntropyLoss(weight=w.to(dev), ignore_index=VOID)
dice = smp.losses.DiceLoss("multiclass", ignore_index=VOID)
for amp in (False, True):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = m(x)
    out = out.float()
    print(f"amp={amp} | finite: {torch.isfinite(out).all().item()} | "
          f"ce {ce(out, y).item():.4f} | dice {dice(out, y).item():.4f}")