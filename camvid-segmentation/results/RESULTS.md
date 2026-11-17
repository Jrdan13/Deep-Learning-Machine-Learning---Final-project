# CamVid Semantic Segmentation for Navigation: Full Results Record

Compiled 2026-11-17 from the training log, `test_metrics.txt` and `nav_metrics.txt`.
Model so far: **U-Net, ResNet34 encoder (ImageNet-pretrained)**. This is the baseline.

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
| Slide deck | https://claude.ai/artifact/Ap2gLsCHBdykahQwHMK7cf |

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

## 8. Experiment table (fill in as runs finish)

| Experiment | Val mIoU | Test mIoU | Road IoU (test) | ROI obstacle to drivable | ROI pavement to road | Car to road | FPS |
|---|---|---|---|---|---|---|---|
| Baseline: U-Net ResNet34 | 0.7617 | 0.6726 | 0.9396 | 5.29% | 9.85% | 4.23% | not measured |
| DeepLabV3+ ResNet34 | | | | | | | |
| Stronger brightness augmentation | | | | | | | |
| Larger input size / random scale crop | | | | | | | |
| U-Net ResNet50 | | | | | | | |
| Lightweight encoder (speed test) | | | | | | | |

## 9. Files

- Code: `src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/nav_metrics.py`, `src/debug_nan.py`
- Results: `results/test_metrics.txt`, `results/nav_metrics.txt`, `results/figures/` (`dataset_check.png`, `test_predictions.png`, `car_as_road.png`, `roi_check.png`)
- Weights (local only): `checkpoints/best.pt`, `checkpoints/last.pt`
- Pushed on first commit: everything except `nav_metrics.py`, `nav_metrics.txt`, `car_as_road.png` and `roi_check.png`. Copy those into the repo clone if you haven't yet.
