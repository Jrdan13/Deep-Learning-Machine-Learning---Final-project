# CamVid segmentation for navigation: method and reproduction guide

Results overview: see the [top-level README](../README.md). Every run in detail: [RESULTS.md](RESULTS.md).

## 1. Task and data

- Dataset: CamVid, 701 labeled street images (360 x 480), SegNet-Tutorial version with 11 classes: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist. Pixels without a label are `Void` (label 11) and are ignored in the loss and in all metrics.
- Splits: train 367, validation 101, test 233. Each split folder has a matching `*annot` folder with the label masks. The test set is used once, for reporting.
- Images are padded from 360 x 480 to 384 x 480 so the networks' downsampling divides evenly. Padded pixels are labeled `Void`.
- Navigation grouping: **Drivable** = Road; **Obstacle** = Building, Tree, Pole, SignSymbol, Fence, Car, Pedestrian, Bicyclist; **Other** = Sky, Pavement.

## 2. Models

All models use a ResNet34 encoder initialized from ImageNet.

| Name | Description |
|---|---|
| `unet` | U-Net decoder with skip connections at every scale (segmentation-models-pytorch). |
| `fcn` | FCN-8s: 1x1 class-score layers on the stride-32, stride-16 and stride-8 encoder features, summed after 2x bilinear upsampling, then upsampled 8x. Written in `src/model.py`. |
| `deeplabv3plus` | DeepLabV3+ from segmentation-models-pytorch with library defaults (dilated encoder, output stride 16, ASPP, light decoder). |

To add a model, add a branch in `build_model()` in `src/model.py` and its name to `ARCHS`.

## 3. Training protocol

- Loss: cross-entropy (`ce`) or cross-entropy + multiclass Dice (`ce_dice`), `Void` ignored. Optional class weights (`--class-weights`); they did not help and are off in all reported comparisons.
- Optimizer: AdamW, learning rate 1e-4 for the encoder and 1e-3 for the rest, weight decay 1e-4, cosine schedule over 40 epochs. Batch size 4, fp32 (fp16 mixed precision gave NaN loss on the T1200 GPU used here).
- Augmentation (`--aug`): `none`; `mild` (flip, brightness/contrast, hue/saturation); `strong` = horizontal flip, random zoom 0.75 to 1.5x with crop or padding (applied with probability 0.8), brightness/contrast, gamma, hue/saturation/value and light blur.
- Seeds 0 and 1. The checkpoint with the best validation mIoU is kept (`checkpoints/<name>/best.pt`).
- Every epoch is logged to `results/runs/<name>/log.csv`; the run settings go to `config.json` and a summary to `summary.txt`.

## 4. Evaluation

- Metrics per class: IoU and Dice from the confusion matrix; mIoU and mean Dice are averages over the 11 classes.
- Navigation measures on the test set:
  - **ROI**: a trapezoid in front of the vehicle (bottom corners at 5% and 95% of the image width on the bottom row, top corners at 38% and 62% of the width at half the image height). Inside it we report the share of obstacle pixels predicted as drivable.
  - **Car called road**: share of Car pixels predicted as Road (whole image).
  - **Pavement called road** inside the ROI (high seed-to-seed variance, not used for conclusions).
- Efficiency: parameters, GFLOPs (a multiply-add counts as 2 FLOPs), peak GPU memory and inference speed at input 1 x 3 x 384 x 480, batch size 1, fp32, model forward pass only (median of 5 timed repeats of 100 iterations). Speed depends strongly on the laptop GPU's power state; `measure.py` prints the GPU clock and power so a throttled GPU is visible (see RESULTS.md section 11.3).

## 5. How to reproduce (Windows PowerShell, from this folder)

Environment:

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Data (about 180 MB download, requires Git):

```
git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial.git data/raw/SegNet-Tutorial
Move-Item data\raw\SegNet-Tutorial\CamVid data\camvid
Remove-Item data\raw\SegNet-Tutorial -Recurse -Force
```

Check the images and augmentations: `python src\dataset.py` (writes `results/figures/aug_check.png`).

Ablation on the U-Net, two seeds (add `--seed 1` and a new `--name` for the second seed; about 20 minutes per run on a laptop T1200):

```
python src\train.py --name base --aug none   --loss ce
python src\train.py --name aug  --aug strong --loss ce
python src\train.py --name loss --aug none   --loss ce_dice
python src\train.py --name both --aug strong --loss ce_dice
python src\train.py --name base_s1 --aug none --loss ce --seed 1
```

Architecture comparison (same recipe):

```
python src\train.py --name fcn_both    --arch fcn           --aug strong --loss ce_dice
python src\train.py --name dlv3p_both  --arch deeplabv3plus --aug strong --loss ce_dice
```

A run can be continued with the same arguments plus `--resume`; an existing run name is never overwritten.

Evaluation, efficiency and figures:

```
python src\eval_run.py --name both      # test metrics -> results/ablation.csv, results/runs/both/
python src\evaluate.py both             # 11-class and 3-group metrics, qualitative figure
python src\nav_metrics.py both          # ROI metrics, car-called-road figure
python src\measure.py                   # parameters, GFLOPs, FPS -> results/efficiency.csv
python src\make_figures.py              # curves, bar charts, flow chart -> results/figures/paper/
```

Results are close to, but not bit-for-bit identical across machines (cuDNN autotuning, GPU type).

## 6. Notes and limitations

- The ablation and the architecture comparison use two seeds per setting; treat differences below about 1 mIoU point as noise.
- The recipe was chosen on the U-Net. DeepLabV3+ seed 0 reached its best validation score at the last epoch, so it may profit from longer training.
- Strong augmentation alone hurt thin classes; with the Dice term the harm disappears.
- Fence IoU drops from about 0.69 on validation to 0.29 to 0.39 on test for every model, so validation and test differ noticeably for that class.
