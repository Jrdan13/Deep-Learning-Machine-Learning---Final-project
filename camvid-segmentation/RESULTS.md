# CamVid Semantic Segmentation for Navigation: Full Results Record

Last updated 2026-11-21. Complete record of every run: setup, training logs of the first model, test and navigation results, the augmentation and loss ablation, and the comparison of U-Net, FCN-8s and DeepLabV3+ (all with a ResNet34 encoder). Per-epoch logs of all later runs are in `results/runs/<name>/log.csv`.

---

## 1. Setup

| Item | Value |
|---|---|
| OS | Windows, PowerShell / VS Code |
| GPU | NVIDIA T1200 Laptop GPU, 4 GB VRAM (driver 573.44, CUDA 12.8) |
| Python | 3.12.10 (venv `.venv`, created with `py -3.12`) |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128 |
| Libraries | segmentation-models-pytorch 0.5.0, albumentations 2.0.8, opencv-python 5.0.0.93, matplotlib 3.11.2, tqdm 4.70.1 |
| Project folder | `Desktop\camvid-segmentation` |
| GitHub | https://github.com/Jrdan13/Deep-Learning-Machine-Learning---Final-project (folder `camvid-segmentation/`, first commit 8daf218) |

## 2. Dataset

- CamVid, SegNet-Tutorial version (11 classes + Void). Void = label 11, ignored in loss and metrics.
- Splits: train 367, val 101, test 233 (each with an `*annot` folder of masks). Images 360x480, padded to 384x480 (padding labeled Void).

Class share of train pixels:

| Class | Share | Class | Share |
|---|---|---|---|
| Sky | 15.79% | Fence | 1.06% |
| Building | 21.80% | Car | 5.50% |
| Pole | 0.92% | Pedestrian | 0.60% |
| Road | 29.68% | Bicyclist | 0.27% |
| Pavement | 4.21% | Void (incl. padding) | 9.95% |
| Tree | 9.12% | SignSymbol | 1.10% |

Navigation grouping: **Drivable** = Road. **Obstacle** = Building, Tree, Pole, SignSymbol, Fence, Car, Pedestrian, Bicyclist. **Other** = Sky, Pavement.

## 3. Model and training settings (baseline)

- Architecture: U-Net, ResNet34 encoder, ImageNet weights, 11 output classes.
- Loss: cross-entropy (class weights = sqrt of median-frequency ratio) + Dice loss; `ignore_index` = Void.
- Optimizer: AdamW, lr 1e-4 (encoder) and 1e-3 (decoder), weight decay 1e-4; CosineAnnealingLR over 40 epochs.
- Batch size 4, 40 epochs, **fp32** (mixed precision off), about 30 s per epoch (about 20 min total).
- Augmentation (train only): horizontal flip p=0.5, random brightness/contrast p=0.5, hue/saturation/value p=0.3, ImageNet normalization.
- Best checkpoint chosen by val mIoU: **epoch 33** (`checkpoints/best.pt`; not in GitHub because of `.gitignore`).
- Problem solved: the first attempt with fp16 mixed precision gave `loss nan` on this GPU. Training in fp32 fixed it.

## 4. Training log (validation set, 101 images)

| Epoch | Loss | Val mIoU | Val Dice | Pixel acc |
|---|---|---|---|---|
| 1 | 1.7473 | 0.4263 | 0.4907 | 0.8154 |
| 2 | 1.1989 | 0.5818 | 0.6809 | 0.8805 |
| 3 | 0.9795 | 0.5831 | 0.6827 | 0.8693 |
| 4 | 0.8475 | 0.6854 | 0.7806 | 0.9225 |
| 5 | 0.7828 | 0.6237 | 0.7308 | 0.8827 |
| 6 | 0.7138 | 0.6734 | 0.7625 | 0.9145 |
| 7 | 0.6383 | 0.6777 | 0.7735 | 0.9250 |
| 8 | 0.5928 | 0.6862 | 0.7822 | 0.9240 |
| 9 | 0.6352 | 0.6897 | 0.7781 | 0.9272 |
| 10 | 0.5581 | 0.7104 | 0.7991 | 0.9341 |
| 11 | 0.5169 | 0.7136 | 0.8016 | 0.9359 |
| 12 | 0.4755 | 0.7226 | 0.8101 | 0.9354 |
| 13 | 0.4568 | 0.7342 | 0.8228 | 0.9364 |
| 14 | 0.4493 | 0.7193 | 0.8068 | 0.9376 |
| 15 | 0.4378 | 0.7251 | 0.8096 | 0.9382 |
| 16 | 0.4304 | 0.7130 | 0.8095 | 0.9353 |
| 17 | 0.4360 | 0.7411 | 0.8284 | 0.9427 |
| 18 | 0.4185 | 0.7328 | 0.8191 | 0.9401 |
| 19 | 0.3932 | 0.6948 | 0.7870 | 0.9330 |
| 20 | 0.3625 | 0.7387 | 0.8249 | 0.9439 |
| 21 | 0.3611 | 0.7285 | 0.8163 | 0.9396 |
| 22 | 0.3619 | 0.7447 | 0.8271 | 0.9464 |
| 23 | 0.3317 | 0.7375 | 0.8229 | 0.9442 |
| 24 | 0.3352 | 0.7337 | 0.8198 | 0.9462 |
| 25 | 0.3267 | 0.7446 | 0.8285 | 0.9485 |
| 26 | 0.3128 | 0.7462 | 0.8310 | 0.9477 |
| 27 | 0.3008 | 0.7577 | 0.8396 | 0.9496 |
| 28 | 0.2937 | 0.7595 | 0.8414 | 0.9491 |
| 29 | 0.2954 | 0.7500 | 0.8344 | 0.9475 |
| 30 | 0.2911 | 0.7386 | 0.8259 | 0.9454 |
| 31 | 0.2784 | 0.7504 | 0.8342 | 0.9485 |
| 32 | 0.2865 | 0.7598 | 0.8404 | 0.9505 |
| **33** | 0.2808 | **0.7617** | 0.8410 | 0.9509 |
| 34 | 0.2769 | 0.7526 | 0.8349 | 0.9487 |
| 35 | 0.2688 | 0.7574 | 0.8378 | 0.9499 |
| 36 | 0.2640 | 0.7583 | 0.8389 | 0.9500 |
| 37 | 0.2625 | 0.7552 | 0.8368 | 0.9501 |
| 38 | 0.2624 | 0.7584 | 0.8387 | 0.9506 |
| 39 | 0.2716 | 0.7582 | 0.8389 | 0.9503 |
| 40 | 0.2654 | 0.7599 | 0.8403 | 0.9505 |

Best validation mIoU: **0.7617 at epoch 33**.

### Per-class validation results at the best epoch

| Class | IoU | Dice |
|---|---|---|
| Sky | 0.9455 | 0.9720 |
| Building | 0.9132 | 0.9547 |
| Pole | 0.1802 | 0.3054 |
| Road | 0.9685 | 0.9840 |
| Pavement | 0.8734 | 0.9324 |
| Tree | 0.9192 | 0.9579 |
| SignSymbol | 0.6651 | 0.7989 |
| Fence | 0.7137 | 0.8329 |
| Car | 0.8562 | 0.9225 |
| Pedestrian | 0.5309 | 0.6936 |
| Bicyclist | 0.8133 | 0.8970 |

## 5. Test results (233 images, `best.pt`)

### 11 classes

| Class | IoU | Dice |
|---|---|---|
| Sky | 0.9203 | 0.9585 |
| Building | 0.8313 | 0.9079 |
| Pole | 0.3182 | 0.4828 |
| Road | 0.9396 | 0.9688 |
| Pavement | 0.8079 | 0.8937 |
| Tree | 0.7625 | 0.8653 |
| SignSymbol | 0.4428 | 0.6138 |
| Fence | 0.4425 | 0.6135 |
| Car | 0.7983 | 0.8878 |
| Pedestrian | 0.5550 | 0.7139 |
| Bicyclist | 0.5803 | 0.7344 |
| **Mean** | **0.6726** | **0.7855** |

Pixel accuracy: 0.9127.

### 3 groups (Drivable / Obstacle / Other)

| Group | IoU | Dice |
|---|---|---|
| Drivable | 0.9396 | 0.9688 |
| Obstacle | 0.9469 | 0.9727 |
| Other | 0.8786 | 0.9354 |
| **Mean** | **0.9217** | **0.9590** |

Pixel accuracy: 0.9615.

### Safety numbers (whole image)

- Obstacle pixels predicted as drivable: 0.42%
- Drivable pixels predicted as obstacle: 0.48%
- Predicted as Road: Car 4.23%, Pedestrian 0.23%, Bicyclist 0.63%

## 6. Navigation metrics (`src/nav_metrics.py`, test set, `best.pt`)

| Metric | Whole image | Inside driving ROI |
|---|---|---|
| Obstacle called drivable | 0.42% | **5.29%** |
| Other (pavement/sky) called drivable | 2.49% | **9.67%** |
| Drivable called not-drivable | 2.99% | 1.44% |
| Pixels called drivable that are not | 3.24% | 2.48% |
| Drivable IoU | 0.9396 | 0.9615 |

- Pavement called Road: 7.06% (whole image), 9.85% (ROI).
- Car called Road: 4.23% (whole image).
- Mean brightness (0-1): all test images 0.389, worst 6 images 0.233.
- Worst Car-to-Road images (car-to-road pixels / car pixels): #27 4165/26960, #28 4157/26823, #25 4007/26752, #29 3983/26630, #22 3943/28048, #31 3937/25323.
- ROI definition (fractions of the 360x480 image): trapezoid with bottom corners at 5% and 95% of the width on the bottom row, top corners at 38% and 62% of the width at 50% of the height. This is a generous corridor that includes some pavement at the bottom corners.

## 7. Observations

1. Large, common classes (Road, Sky, Building, Pavement, Car) reach 0.80+ IoU on test. Thin or small classes are weakest: Pole 0.32, SignSymbol 0.44, Fence 0.44, Pedestrian 0.56, Bicyclist 0.58.
2. Val mIoU (0.762) is higher than test mIoU (0.673). Val was used to pick the epoch, and the test sequences are harder.
3. Whole-image safety numbers hide the risk: obstacle-to-drivable leakage is about 12 times higher inside the driving ROI (5.29% vs 0.42%).
4. Pavement is confused with Road, mostly near curbs.
5. The worst Car-to-Road frames (#22-#31) are consecutive frames of one dark scene. The error is a band along the bottom edge of the car, so the car is detected but its ground footprint is cut short. This is one failure case, not six independent ones.
6. CamVid test frames come from a few video sequences, so neighboring frames repeat the same mistakes. Treat differences under about 1 mIoU point between experiments as possibly noise.

## 8. Experiment table (superseded by section 10)

| Experiment | Val mIoU | Test mIoU | Road IoU (test) | ROI obstacle to drivable | ROI pavement to road | Car to road | FPS |
|---|---|---|---|---|---|---|---|
| Baseline: U-Net ResNet34 | 0.7617 | 0.6726 | 0.9396 | 5.29% | 9.85% | 4.23% | not measured |
| DeepLabV3+ ResNet34 | | | | | | | |
| Stronger brightness augmentation | | | | | | | |
| Larger input size / random scale crop | | | | | | | |
| U-Net ResNet50 | | | | | | | |
| Lightweight encoder (speed test) | | | | | | | |

## 9. Files

- Code (`src/`): `dataset.py`, `model.py`, `train.py`, `eval_run.py`, `evaluate.py`, `nav_metrics.py`, `measure.py`, `make_figures.py`
- Per-run outputs (`results/runs/<name>/`): `log.csv`, `config.json`, `summary.txt`, `test_per_class.txt`
- Summary tables: `results/ablation.csv`, `results/efficiency.csv`
- Figures: `results/figures/` and `results/figures/paper/` (PNG and PDF)
- Model weights are not in the repository (`checkpoints/` is excluded by `.gitignore`).

---

## 10. Ablation study: augmentation and loss (U-Net ResNet34, 40 epochs, 2 seeds)

Design: 2x2 factorial. Augmentation: none vs strong (random zoom 0.75-1.5x with crop/pad, flip, brightness/contrast, gamma, hue/saturation/value, light blur). Loss: cross-entropy (CE) vs CE + Dice. No class weights. Seeds 0 and 1. Per-epoch logs are in `results/runs/<name>/log.csv`. "val last-5" is the mean validation mIoU of the last 5 epochs. Test results use `best.pt` (best validation mIoU).

### 10.1 All runs (as in results/ablation.csv)

| name | aug | loss | val best mIoU | val last-5 mIoU | test mIoU | test Dice | road IoU | ROI obstacle to drivable % | ROI pavement to road % | car to road % |
|---|---|---|---|---|---|---|---|---|---|---|
| current (reference) | mild | CE+Dice+class weights | 0.7617 | 0.7580 | 0.6726 | 0.7855 | 0.9396 | 5.29 | 9.85 | 4.23 |
| base | none | CE | 0.7289 | 0.7248 | 0.6465 | 0.7625 | 0.9235 | 5.99 | 11.04 | 4.72 |
| base_s1 | none | CE | 0.7410 | 0.7305 | 0.6499 | 0.7630 | 0.9476 | 5.37 | 5.54 | 4.30 |
| aug | strong | CE | 0.7100 | 0.7059 | 0.6313 | 0.7433 | 0.9439 | 3.57 | 9.95 | 2.79 |
| aug_s1 | strong | CE | 0.7265 | 0.7235 | 0.6419 | 0.7546 | 0.9450 | 3.78 | 11.17 | 2.94 |
| loss | none | CE+Dice | 0.7514 | 0.7448 | 0.6647 | 0.7769 | 0.9453 | 4.91 | 8.12 | 3.73 |
| loss_s1 | none | CE+Dice | 0.7609 | 0.7583 | 0.6636 | 0.7753 | 0.9489 | 4.70 | 7.51 | 3.71 |
| both | strong | CE+Dice | 0.7564 | 0.7539 | 0.6748 | 0.7836 | 0.9496 | 4.76 | 6.83 | 3.77 |
| both_s1 | strong | CE+Dice | 0.7583 | 0.7558 | 0.6632 | 0.7749 | 0.9498 | 3.59 | 7.14 | 2.77 |
| both_w | strong | CE+Dice+class weights | 0.7512 | 0.7460 | 0.6581 | 0.7711 | 0.9344 | 4.92 | 6.22 | 3.64 |

`current` was trained before the run logger existed (mild augmentation, class-weighted CE + Dice); its validation numbers come from the training printout. `both_w` has one seed only.

### 10.2 Two-seed means (range in brackets)

| recipe | val last-5 mIoU | test mIoU | test Dice | road IoU | ROI obstacle to drivable % | ROI pavement to road % | car to road % |
|---|---|---|---|---|---|---|---|
| base (none, CE) | 0.728 [0.725-0.731] | 0.648 [0.647-0.650] | 0.763 | 0.936 [0.924-0.948] | 5.68 [5.37-5.99] | 8.29 [5.54-11.04] | 4.51 [4.30-4.72] |
| aug (strong, CE) | 0.715 [0.706-0.724] | 0.637 [0.631-0.642] | 0.749 | 0.945 [0.944-0.945] | 3.67 [3.57-3.78] | 10.56 [9.95-11.17] | 2.87 [2.79-2.94] |
| loss (none, CE+Dice) | 0.752 [0.745-0.758] | 0.664 [0.664-0.665] | 0.776 | 0.947 [0.945-0.949] | 4.80 [4.70-4.91] | 7.82 [7.51-8.12] | 3.72 [3.71-3.73] |
| both (strong, CE+Dice) | 0.755 [0.754-0.756] | 0.669 [0.663-0.675] | 0.779 | 0.950 [0.950-0.950] | 4.17 [3.59-4.76] | 6.99 [6.83-7.14] | 3.27 [2.77-3.77] |

### 10.3 Main effects on test mIoU (percentage points, mean of 2 seeds)

| change | without the other factor | with the other factor |
|---|---|---|
| add Dice to CE | +1.6 (base to loss) | +3.2 (aug to both) |
| add strong augmentation | -1.2 (base to aug) | +0.5 (loss to both) |

Validation (last-5 mean) shows the same pattern: Dice +2.4 / +4.0, augmentation -1.3 / +0.3.

### 10.4 Test per-class IoU

| class | base | base_s1 | aug | aug_s1 | loss | loss_s1 | both | both_s1 | both_w | current |
|---|---|---|---|---|---|---|---|---|---|---|
| Sky | 0.9244 | 0.9235 | 0.9213 | 0.9229 | 0.9229 | 0.9224 | 0.9179 | 0.9179 | 0.9173 | 0.9203 |
| Building | 0.8276 | 0.8254 | 0.8162 | 0.8107 | 0.8271 | 0.8220 | 0.8315 | 0.8154 | 0.8148 | 0.8313 |
| Pole | 0.2876 | 0.2933 | 0.2182 | 0.2489 | 0.2989 | 0.2958 | 0.2853 | 0.2910 | 0.2999 | 0.3182 |
| Road | 0.9235 | 0.9476 | 0.9439 | 0.9450 | 0.9453 | 0.9489 | 0.9496 | 0.9498 | 0.9344 | 0.9396 |
| Pavement | 0.7619 | 0.8257 | 0.8045 | 0.8100 | 0.8195 | 0.8164 | 0.8357 | 0.8255 | 0.8032 | 0.8079 |
| Tree | 0.7616 | 0.7540 | 0.7544 | 0.7485 | 0.7565 | 0.7558 | 0.7544 | 0.7526 | 0.7515 | 0.7625 |
| SignSymbol | 0.4268 | 0.4142 | 0.4181 | 0.4309 | 0.4351 | 0.4205 | 0.4554 | 0.4390 | 0.4542 | 0.4428 |
| Fence | 0.3974 | 0.3727 | 0.3173 | 0.3331 | 0.4020 | 0.3687 | 0.3877 | 0.3561 | 0.3179 | 0.4425 |
| Car | 0.8095 | 0.8214 | 0.8335 | 0.8343 | 0.8210 | 0.8126 | 0.8589 | 0.8121 | 0.8141 | 0.7983 |
| Pedestrian | 0.5358 | 0.4910 | 0.4703 | 0.5026 | 0.5739 | 0.5678 | 0.5539 | 0.5544 | 0.5537 | 0.5550 |
| Bicyclist | 0.4549 | 0.4795 | 0.4467 | 0.4740 | 0.5098 | 0.5685 | 0.5918 | 0.5809 | 0.5787 | 0.5803 |

### 10.5 Findings

1. **Dice + CE is the main improvement.** Test mIoU gain of +1.6 points (no augmentation) and +3.2 points (with augmentation); the direction held in both seeds and on validation.
2. **Strong augmentation alone hurts** test mIoU (-1.2 points) and the thin classes (Pole 0.29 to 0.23, Fence 0.39 to 0.33 on average). With Dice the harm disappears and the net effect is +0.5 points, which is smaller than the seed-to-seed spread (up to 1.2 points for `both`). Its contribution on top of Dice is therefore not established.
3. **Dice, not augmentation, produced the gains on Pedestrian and Bicyclist** (test IoU: Pedestrian 0.513 to 0.571, Bicyclist 0.467 to 0.539 from base to loss).
4. **Navigation safety:** strong augmentation alone lowered obstacle-to-drivable leakage in the ROI (5.68% to 3.67%) and car-to-road (4.51% to 2.87%), with non-overlapping seed ranges. Combined with Dice the effect is smaller and the ranges overlap.
5. **Pavement-to-road in the ROI is not a reliable metric across seeds** (CE-only runs ranged 5.54% to 11.17%; Dice runs 6.83% to 8.12%). An earlier remark that Dice or augmentation reduced it from 11.04% to 6.83% came from a single baseline seed and should not be reported.
6. **Class weights did not help** (`both_w`: test 0.6581, val last-5 0.7460, Fence 0.318), so the final recipe uses none.
7. **Unsolved:** Pole stays about 0.29 and Fence about 0.37 in every run.

### 10.6 Caveats

- Two seeds per cell is too few for significance tests; report means with ranges.
- The validation set has 101 images and the test frames come from a few video sequences (neighboring frames repeat the same errors), so differences below about 1 mIoU point are not reliable.
- ROI percentages are computed on small pixel counts and the ROI shape is a hand-chosen trapezoid.
- The checkpoint is chosen by validation mIoU; test is used for reporting only.

### 10.7 See also

The architecture comparison (U-Net, FCN-8s, DeepLabV3+) is in section 11.

---

## 11. Model comparison: U-Net vs FCN-8s vs DeepLabV3+ (same encoder, same recipe, 2 seeds)

**Models.** All three use the same ResNet34 encoder (ImageNet-pretrained). U-Net: decoder with skip connections at every scale. FCN-8s: 1x1 class-score layers on the stride-32, stride-16 and stride-8 features, summed after 2x bilinear upsampling, then upsampled 8x. DeepLabV3+: segmentation-models-pytorch implementation with library defaults (dilated encoder, output stride 16, ASPP module, light decoder using stride-4 features). Recipe for all: strong augmentation, CE + Dice, no class weights, AdamW, cosine schedule, 40 epochs, batch 4, fp32, seeds 0 and 1, checkpoint chosen by validation mIoU.

### 11.1 Accuracy

| model | seed | val last-5 mIoU | test mIoU | test Dice | road IoU | ROI obstacle to drivable % | ROI pavement to road % | car to road % |
|---|---|---|---|---|---|---|---|---|
| U-Net (`both`) | 0 | 0.7539 | 0.6748 | 0.7836 | 0.9496 | 4.76 | 6.83 | 3.77 |
| U-Net (`both_s1`) | 1 | 0.7558 | 0.6632 | 0.7749 | 0.9498 | 3.59 | 7.14 | 2.77 |
| **U-Net mean** | | **0.7549** | **0.6690** | **0.7793** | **0.9497** | **4.17** | **6.99** | **3.27** |
| FCN-8s (`fcn_both`) | 0 | 0.7248 | 0.6511 | 0.7639 | 0.9329 | 5.50 | 14.21 | 4.27 |
| FCN-8s (`fcn_both_s1`) | 1 | 0.7358 | 0.6497 | 0.7616 | 0.9450 | 5.03 | 8.72 | 3.92 |
| **FCN mean** | | **0.7303** | **0.6504** | **0.7628** | **0.9390** | **5.27** | **11.47** | **4.09** |
| DeepLabV3+ (`dlv3p_both`) | 0 | 0.7537 | 0.6427 | 0.7560 | 0.9434 | 4.94 | 10.41 | 3.90 |
| DeepLabV3+ (`dlv3p_both_s1`) | 1 | 0.7502 | 0.6464 | 0.7585 | 0.9427 | 5.16 | 9.50 | 4.10 |
| **DeepLabV3+ mean** | | **0.7520** | **0.6446** | **0.7573** | **0.9431** | **5.05** | **9.96** | **4.00** |

Best validation mIoU: U-Net 0.7564 / 0.7583; FCN 0.7332 (epoch 28) / 0.7382 (epoch 30); DeepLabV3+ 0.7552 (epoch 40) / 0.7512 (epoch 33).

Validation-to-test drop in mIoU (means): U-Net 75.5 to 66.9 (-8.6 points), FCN 73.0 to 65.0 (-8.0), DeepLabV3+ 75.2 to 64.5 (-10.7).

### 11.2 Test per-class IoU (two-seed means)

| class | U-Net | FCN-8s | DeepLabV3+ |
|---|---|---|---|
| Sky | 0.9179 | 0.9029 | 0.9134 |
| Building | 0.8235 | 0.8161 | 0.8132 |
| Pole | 0.2882 | 0.2140 | 0.2654 |
| Road | 0.9497 | 0.9390 | 0.9431 |
| Pavement | 0.8306 | 0.7950 | 0.8069 |
| Tree | 0.7535 | 0.7525 | 0.7469 |
| SignSymbol | 0.4472 | 0.4245 | 0.4260 |
| Fence | 0.3719 | 0.3898 | 0.2920 |
| Car | 0.8355 | 0.8172 | 0.8118 |
| Pedestrian | 0.5542 | 0.5206 | 0.5288 |
| Bicyclist | 0.5864 | 0.5832 | 0.5428 |

Fence IoU on validation vs test (seed 0 / seed 1): U-Net 0.690 / 0.694 vs 0.388 / 0.356; FCN 0.666 / 0.659 vs 0.404 / 0.376; DeepLabV3+ 0.680 / 0.694 vs 0.299 / 0.285.

### 11.3 Efficiency

Parameters, GFLOPs and peak memory (identical in both measurement runs; input 1x3x384x480, GFLOPs counted with torch FlopCounterMode, a multiply-add counts as 2 FLOPs):

| model | parameters (M) | GFLOPs | peak memory (MB) | speed relative to U-Net |
|---|---|---|---|---|
| U-Net | 24.44 | 44.5 | 165 | 1.00 |
| FCN-8s | 21.29 | 26.9 | 128 | 1.75 |
| DeepLabV3+ | 22.44 | 44.4 | 162 | 0.94 |

**Absolute FPS was measured while the GPU was power-limited and is not reported as a result.** Three measurement runs on the same laptop gave different answers: 2026-11-20 (U-Net and FCN only) 45.5 and 79.6 FPS; 2026-11-21 after the DeepLabV3+ training 6.3 (U-Net), 11.1 (FCN) and 5.9 (DeepLabV3+) FPS; and 2026-11-21 16:08 with the updated `measure.py` 6.4, 11.3 and 6.0 FPS, stable across 5 repeats. During the last run `nvidia-smi` showed the GPU streaming-multiprocessor clock at 300 MHz (maximum 2100 MHz) at 100% utilization, with the power cap reading 15 W, against 30 W at the start of the project. The clock ratio (2100 / 300 = 7) matches the roughly 7-fold slowdown, and the ratios between the models were the same in every run (FCN 1.75 times and DeepLabV3+ 0.94 times the speed of U-Net). The slowdown is therefore a laptop power or clock limit, not a property of the models. GFLOPs and relative speeds are reported; absolute FPS has to be re-measured with the GPU at normal clocks before it is quoted. `results/efficiency.csv` currently holds the throttled values.

### 11.4 Findings

1. **U-Net is the most accurate on test** (0.669 vs FCN 0.650 vs DeepLabV3+ 0.645) and is best on all navigation measures: road IoU 0.950 vs 0.939 / 0.943, obstacle-to-drivable in the ROI 4.17% vs 5.27% / 5.05%, car-to-road 3.27% vs 4.09% / 4.00%. Test seed ranges do not overlap between U-Net and the other two.
2. **FCN and DeepLabV3+ are close on test** (difference 0.6 points, inside the noise) but they rank differently on validation: DeepLabV3+ (0.752) matches U-Net (0.755) and beats FCN (0.730) by 2.2 points on validation, then drops to last place on test. Its validation-to-test drop (-10.7 points) is the largest, and Fence explains much of it (0.69 on validation, 0.29 on test).
3. **Per class:** DeepLabV3+ is better than FCN on Pole (+5.1 points) but worse on Fence (-9.8) and Bicyclist (-4.0); against U-Net it is lower on Fence (-8.0), Bicyclist (-4.4), Pedestrian (-2.5), Pavement and Car (-2.4), Pole (-2.3). Poles stay at 0.21 to 0.29 for every architecture.
4. **Cost:** DeepLabV3+ needs the same compute as U-Net (44.4 vs 44.5 GFLOPs) and runs at about 0.94 of its speed, so U-Net gives better accuracy at no extra cost. FCN uses 40% less compute and is about 1.75 times faster, so it is the speed option, at a cost of about 1.9 mIoU points against U-Net.
5. The pavement-to-road numbers (U-Net 6.99%, DeepLabV3+ 9.96%, FCN 11.47%) point the same way but this metric varied widely across seeds in the ablation and should not be relied on.

### 11.5 Caveats

- One shared recipe was used for all architectures. It was chosen from the U-Net ablation, so FCN and DeepLabV3+ might prefer different settings (learning rates, weights, augmentation).
- DeepLabV3+ seed 0 reached its best validation mIoU at the last epoch (40), so it may still be improving at the 40-epoch budget. It used the library defaults.
- Two seeds per model is too few for significance tests. Validation has 101 images and test frames come from a few sequences, so differences below about 1 point are unreliable, and the disagreement between validation and test for DeepLabV3+ shows how much the ranking can move.
- Absolute FPS is pending (see 11.3).

### 11.6 Open items

- Re-measure absolute FPS with the GPU at normal clocks (`src/measure.py`), then regenerate `fig_speed_accuracy`.
- Demo video.
