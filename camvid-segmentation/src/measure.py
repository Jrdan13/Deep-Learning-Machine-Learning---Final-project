"""Efficiency measurement: parameters, GFLOPs, FPS (batch 1), peak GPU memory for every architecture.
Repeats the timing 5 times per model and logs GPU clocks and power, so an unstable or throttled GPU
is visible. Run it with the laptop plugged in, power mode on best performance, nothing else using the GPU."""
import csv
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import build_model, ARCHS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "efficiency.csv"
COLS = ["arch", "params_M", "GFLOPs", "fps_batch1", "fps_max", "peak_mem_MB"]
REPEATS, ITERS = 5, 100


def gpu_state():
    q = "clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,pstate"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:
        return f"nvidia-smi unavailable ({e})"


def main():
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU state before (sm clock, max sm clock, power draw, power limit, temp, util, pstate):")
    print("   ", gpu_state(), "\n")
    x = torch.randn(1, 3, 384, 480, device=dev)
    rows = []
    for arch in ARCHS:
        model = build_model(arch, pretrained=False).to(dev).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                model(x)
            gflops = fc.get_total_flops() / 1e9  # a multiply-add counts as 2 FLOPs
            for _ in range(30):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fps, state = [], ""
            for k in range(REPEATS):
                t = time.time()
                for _ in range(ITERS):
                    model(x)
                torch.cuda.synchronize()
                fps.append(ITERS / (time.time() - t))
                if k == REPEATS - 2:
                    state = gpu_state()
            mem = torch.cuda.max_memory_allocated() / 2 ** 20
        med, mx = statistics.median(fps), max(fps)
        flag = "   <-- UNSTABLE: check power mode and background GPU use" if min(fps) < 0.8 * mx else ""
        print(f"{arch:<14} FPS median {med:.1f}, max {mx:.1f}, runs {[round(f, 1) for f in fps]}{flag}")
        print("   GPU during test:", state)
        rows.append([arch, f"{params:.2f}", f"{gflops:.1f}", f"{med:.1f}", f"{mx:.1f}", f"{mem:.0f}"])
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(COLS)
        wr.writerows(rows)
    print("\n| " + " | ".join(COLS) + " |")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\nSaved", OUT)


if __name__ == "__main__":
    main()