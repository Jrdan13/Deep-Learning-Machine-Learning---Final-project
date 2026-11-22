"""Demo video: runs trained segmentation models on a street video (or on CamVid test frames) and
writes an MP4 with the navigation map (green = drivable, red = obstacle, gray = other).

Examples (from the project folder):
  python src\\demo.py --video clip.mp4                       # 2x2 grid: original, U-Net, FCN-8s, DeepLabV3+
  python src\\demo.py --video clip.mp4 --names both_s1      # one model only
  python src\\demo.py --camvid --start 0 --count 80          # CamVid test frames with ground truth
Models are loaded from checkpoints/<name>/best.pt; the architecture comes from results/runs/<name>/config.json."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "camvid"
CKPT = ROOT / "checkpoints"
RUNS = ROOT / "results" / "runs"

# class -> navigation group: 0 drivable (Road), 1 obstacle, 2 other (Sky, Pavement); 3 = Void (ground truth only)
GROUP = np.full(12, 1, np.uint8)
GROUP[3] = 0
GROUP[[0, 4]] = 2
GROUP[11] = 3
NAV_RGB = np.array([[0, 200, 0], [220, 30, 30], [140, 140, 140], [25, 25, 25]], np.uint8)
ARCH_LABEL = {"unet": "U-Net", "fcn": "FCN-8s", "deeplabv3plus": "DeepLabV3+"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MODEL_H = 384  # network input height for videos (width follows the aspect ratio, multiple of 32)


# ---------------------------------------------------------------- model
class Predictor:
    def __init__(self, name, smooth, device=None):
        import torch
        from model import build_model
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = RUNS / name / "config.json"
        arch = json.loads(cfg.read_text()).get("arch", "unet") if cfg.exists() else "unet"
        self.model = build_model(arch, pretrained=False).to(self.dev)
        self.model.load_state_dict(torch.load(CKPT / name / "best.pt", map_location=self.dev))
        self.model.eval()
        self.label = ARCH_LABEL.get(arch, arch)
        self.smooth, self.prev, self.fps = smooth, None, None
        self.mean = torch.tensor(MEAN, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.dev).view(1, 3, 1, 1)
        with torch.no_grad():  # warm-up
            for _ in range(3):
                self.model(torch.zeros(1, 3, 384, 480, device=self.dev))

    def reset(self):
        self.prev = None

    def predict(self, rgb):
        """rgb: uint8 HxWx3 at the size the network should see. Returns a navigation-group map (HxW uint8)."""
        torch = self.torch
        x = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self.mean) / self.std
        with torch.no_grad():
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            p = self.model(x).softmax(1)
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t
            if self.smooth > 0 and self.prev is not None and self.prev.shape == p.shape:
                p = self.smooth * self.prev + (1 - self.smooth) * p
            self.prev = p
            cls = p.argmax(1)[0].cpu().numpy().astype(np.uint8)
        self.fps = 1.0 / dt if self.fps is None else 0.9 * self.fps + 0.1 / dt
        return GROUP[cls]


# ---------------------------------------------------------------- drawing
def nav_overlay(rgb, group_map, alpha=0.45):
    h, w = rgb.shape[:2]
    g = cv2.resize(group_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return (rgb * (1 - alpha) + NAV_RGB[g] * alpha).astype(np.uint8)


def put_label(img, text, org=(10, 26), scale=0.6):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def legend_strip(width, note):
    bar = np.full((34, width, 3), 20, np.uint8)
    x = 10
    for color, text in zip(NAV_RGB[:3], ["Drivable", "Obstacle", "Other (sky, pavement)"]):
        cv2.rectangle(bar, (x, 9), (x + 16, 25), tuple(int(c) for c in color), -1)
        cv2.putText(bar, text, (x + 22, 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += 22 + cv2.getTextSize(text, FONT, 0.5, 1)[0][0] + 24
    if x + 150 < width:
        cv2.putText(bar, note, (x + 10, 22), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def make_grid(panels, cols):
    h, w = panels[0].shape[:2]
    rows = -(-len(panels) // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    return canvas


def title_card(width, height, lines):
    card = np.zeros((height, width, 3), np.uint8)
    y = height // 2 - 20 * (len(lines) - 1)
    for k, text in enumerate(lines):
        scale = 1.0 if k == 0 else 0.7
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2 if k == 0 else 1)
        cv2.putText(card, text, ((width - tw) // 2, y + 40 * k), FONT, scale, (255, 255, 255),
                    2 if k == 0 else 1, cv2.LINE_AA)
    return card


# ---------------------------------------------------------------- sources: yield dicts
def to_model_input(bgr):
    h, w = bgr.shape[:2]
    nw = max(32, int(round(w * MODEL_H / h / 32)) * 32)
    return cv2.cvtColor(cv2.resize(bgr, (nw, MODEL_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def video_source(path, max_seconds, every):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_seconds and i >= max_seconds * fps):
                break
            if i % every == 0:
                yield {"show": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "model": to_model_input(frame),
                       "valid": None, "gt": None}
            i += 1
        cap.release()
    return fps / every, gen()


def camvid_source(split, start, count, fps):
    imgs = sorted((DATA / split).glob("*.png"))
    masks = sorted((DATA / f"{split}annot").glob("*.png"))
    if not imgs:
        raise SystemExit(f"No CamVid images found in {DATA / split}")

    def gen():
        for i in range(start, min(start + count, len(imgs))):
            bgr = cv2.imread(str(imgs[i]))
            mask = cv2.imread(str(masks[i]), cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            padded = cv2.copyMakeBorder(rgb, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_CONSTANT, value=0)
            yield {"show": rgb, "model": padded, "valid": (h, w), "gt": GROUP[np.minimum(mask, 11)]}
    return fps, gen()


# ---------------------------------------------------------------- main loop
def run(source, fps, predictors, out_path, panel_width=640, title=True, note="", snap_at=40):
    writer, frames, snap_saved, size = None, 0, False, None
    for p in predictors:
        p.reset()
    t0 = time.time()
    for item in source:
        show = item["show"]
        pw = panel_width
        ph = int(round(pw * show.shape[0] / show.shape[1]))
        panels = [show]
        labels = ["Original"]
        if item["gt"] is not None:
            panels.append(nav_overlay(show, item["gt"]))
            labels.append("Ground truth")
        for p in predictors:
            g = p.predict(item["model"])
            if item["valid"] is not None:
                g = g[:item["valid"][0], :item["valid"][1]]
            panels.append(nav_overlay(show, g))
            labels.append(f"{p.label}  {p.fps:.0f} FPS (model only)")
        panels = [cv2.resize(x, (pw, ph), interpolation=cv2.INTER_AREA) for x in panels]
        for x, text in zip(panels, labels):
            put_label(x, text)
        cols = 1 if len(panels) == 1 else (2 if len(panels) <= 4 else 3)
        grid = make_grid(panels, cols)
        frame = np.vstack([grid, legend_strip(grid.shape[1], note)])
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise SystemExit("Could not open the video writer (mp4v codec missing?).")
            if title:
                card = title_card(size[0], size[1], [
                    "Semantic segmentation for navigation",
                    "Models trained on CamVid (U-Net, FCN-8s, DeepLabV3+, ResNet34 encoder)",
                    "Green = drivable   Red = obstacle   Gray = other (sky, pavement)"])
                for _ in range(int(fps * 3)):
                    writer.write(card)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if frames == snap_at:
            cv2.imwrite(str(out_path.with_name("demo_snapshot.png")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            snap_saved = True
        frames += 1
        if frames % 30 == 0:
            print(f"  {frames} frames, {frames / (time.time() - t0):.1f} frames/s overall")
    if writer is None:
        raise SystemExit("No frames were read.")
    if not snap_saved:
        cv2.imwrite(str(out_path.with_name("demo_snapshot.png")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved {out_path} ({frames} frames, {size[0]}x{size[1]}, {fps:.1f} fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="input video file (mp4, avi, mov)")
    ap.add_argument("--camvid", action="store_true", help="use CamVid test frames (with ground truth) instead of a video")
    ap.add_argument("--start", type=int, default=0, help="first CamVid frame")
    ap.add_argument("--count", type=int, default=80, help="number of CamVid frames")
    ap.add_argument("--camvid-fps", type=float, default=4.0, help="playback speed for CamVid frames")
    ap.add_argument("--names", nargs="+", default=["both_s1", "fcn_both_s1", "dlv3p_both"],
                    help="run names to show (best validation checkpoint of each architecture by default)")
    ap.add_argument("--out", default=str(ROOT / "results" / "demo" / "demo.mp4"))
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--max-seconds", type=float, default=30.0, help="stop after this many seconds of video (0 = all)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th video frame")
    ap.add_argument("--smooth", type=float, default=0.0, help="temporal smoothing 0..0.9 of class probabilities (demo only)")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    if args.camvid == bool(args.video):
        raise SystemExit("Give exactly one of --video <file> or --camvid.")
    fps, source = (camvid_source("test", args.start, args.count, args.camvid_fps) if args.camvid
                   else video_source(args.video, args.max_seconds, args.every))
    predictors = [Predictor(n, args.smooth) for n in args.names]
    note = "Trained on CamVid." + (f" Temporal smoothing {args.smooth} (demo only)." if args.smooth > 0 else "")
    run(source, fps, predictors, Path(args.out), args.panel_width, not args.no_title, note)


if __name__ == "__main__":
    main()
