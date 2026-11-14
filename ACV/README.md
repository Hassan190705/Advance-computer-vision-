# Reproducible Semantic Segmentation Experiment
**Assignment B: Build a Reproducible Segmentation Experiment**  
*Advanced Computer Vision | CS107203CS | Unit 3*

This repository contains an end-to-end, fully reproducible semantic segmentation experiment built with PyTorch. It benchmarks a modest encoder–decoder against a controlled architectural modification (**Receptive Field Dilation in the Bottleneck**) on the public **CamVid (11-class)** road scene dataset under a strictly matched parameter budget (0.00% parameter delta).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Environment & Prerequisites](#environment--prerequisites)
4. [Quick Start: One-Command Reproduction](#quick-start-one-command-reproduction)
5. [Step-by-Step Usage Guide](#step-by-step-usage-guide)
   - [1. Data Contract Verification](#1-data-contract-verification)
   - [2. Model Training](#2-model-training)
   - [3. Evaluation on Test Set](#3-evaluation-on-test-set)
   - [4. Latency Benchmarking](#4-latency-benchmarking)
   - [5. Visualizations & Failure Case Analysis](#5-visualizations--failure-case-analysis)
6. [Data Contract Specifications](#data-contract-specifications)
7. [Architectures & Controlled Change Mechanism](#architectures--controlled-change-mechanism)
8. [Summary of Empirical Results](#summary-of-empirical-results)
9. [What the Experiment Does NOT Establish](#what-the-experiment-does-not-establish)
10. [Submission & Citations](#submission--citations)

---

## Project Overview

The project fulfills all four tasks specified in Assignment B:
1. **Data Contract:** Uses verified, human-annotated CamVid data (701 frames: 367 train / 101 val / 233 test), 11 active classes, and an explicit ignored label (`255`).
2. **Baseline Model:** Modest 4-stage lightweight U-Net encoder–decoder (541,003 parameters), deterministic seed (`42`), AdamW optimizer, and cosine annealing scheduler.
3. **Controlled Change:** Evaluates dilated convolutions in the bottleneck ($d=2, 4$) while keeping the parameter budget strictly matched (541,003 vs 541,003; $\Delta = 0$).
4. **Evidence & Documentation:** Reports per-class IoU, mIoU, parameter count, measured CPU latency (batch size = 1), qualitative visual comparisons, and explicit failure case analysis.

---

## Directory Structure

```text
ACV/
├── README.md                      # Project documentation and utilization guide
├── REPORT.md                      # Comprehensive academic report & discussion
├── run_all_experiments.py         # Master one-command reproduction runner
├── train.py                       # Modular training pipeline with logging & stopping rules
├── evaluate.py                    # Exact evaluation script (mIoU, per-class IoU, latency)
├── visualize.py                   # Generates comparative prediction panels & error maps
├── requirements.txt               # Pinned dependencies
├── data/
│   └── CamVid/                    # CamVid 11-class benchmark dataset
│       ├── train/                 # 367 training images
│       ├── trainannot/            # 367 training masks (labels 0..11)
│       ├── val/                   # 101 validation images
│       ├── valannot/              # 101 validation masks
│       ├── test/                  # 233 test images
│       └── testannot/             # 233 test masks
├── src/
│   ├── data_contract.py           # Contract definition, color palettes, transforms & verifier
│   ├── models.py                  # Baseline, Dilated Bottleneck & Additive Skip architectures
│   └── metrics.py                 # Streaming confusion matrix, per-class IoU, pixel accuracy
├── experiments/                   # Checkpoints, structured logs, and JSON outputs
│   ├── consolidated_results.json  # Consolidated experimental metrics across models
│   ├── baseline_seed42_res96x128_bc16/
│   │   ├── best_model.pth         # Best baseline checkpoint (highest val mIoU)
│   │   ├── last_model.pth         # Final epoch checkpoint
│   │   ├── training.log           # Plaintext training execution log
│   │   ├── history.json           # Epoch-by-epoch loss and validation metrics
│   │   └── test_eval_results.json # Official test split evaluation metrics
│   └── dilated_bottleneck_seed42_res96x128_bc16/
│       ├── best_model.pth         # Best dilated variant checkpoint
│       ├── last_model.pth         # Final epoch checkpoint
│       ├── training.log           # Plaintext training execution log
│       ├── history.json           # Epoch-by-epoch metrics
│       └── test_eval_results.json # Official test split evaluation metrics
└── visualizations/
    ├── comparative_predictions.png# Side-by-side [Input | Ground Truth | Baseline | Dilated]
    └── failure_case_analysis.png  # Explicit error map highlighting model breakdown modes
```

---

## Environment & Prerequisites

### Hardware
* Runs seamlessly on **CPU** (e.g. dual-core Intel Core i5 with integrated graphics) or **GPU** (NVIDIA CUDA).
* All data preloads into RAM (~70 MB) for sub-minute epoch times on standard laptops.

### Software
* Python `3.10+` (tested on Python `3.14.3`)
* PyTorch `2.0+`
* Torchvision `0.15+`

Install the required packages:
```bash
pip install torch torchvision matplotlib pandas tabulate pillow
```
Or with `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Quick Start: One-Command Reproduction

To execute the entire automated experiment pipeline from start to finish, run:

```bash
python run_all_experiments.py
```

### What this command does:
1. Verifies the dataset schema, label integrity, and split counts.
2. Trains the **Baseline Model** for 5 epochs with fixed seed `42`.
3. Trains the **Dilated Bottleneck Variant** for 5 epochs under identical conditions.
4. Evaluates both trained checkpoints on the **233 test images**.
5. Measures CPU inference latency over 30 warm-up and 100 timed passes.
6. Generates high-resolution comparative visualization panels and failure case analysis.
7. Prints consolidated summary tables to the terminal and saves all metrics to `experiments/consolidated_results.json`.

---

## Step-by-Step Usage Guide

If you prefer to run individual modules separately:

### 1. Data Contract Verification
Verify file counts, split partitions, integer label distributions, and color schemas:
```bash
python src/data_contract.py
```
*Expected output:* Confirms 367 train, 101 val, and 233 test pairs with labels in range `0..11`.

---

### 2. Model Training
Train any supported architecture with custom hyperparameters:

```bash
# Train Baseline (Standard Convolutions, d=1)
python train.py --mechanism baseline --epochs 5 --batch_size 16 --base_channels 16 --resolution 96 128 --seed 42

# Train Controlled Variant (Dilated Bottleneck, d=2 and d=4)
python train.py --mechanism dilated_bottleneck --epochs 5 --batch_size 16 --base_channels 16 --resolution 96 128 --seed 42

# Train Additive Skips Variant (Residual-style skip fusion)
python train.py --mechanism additive_skips --epochs 5 --batch_size 16 --base_channels 16 --resolution 96 128 --seed 42
```

#### Key Arguments:
* `--mechanism`: `baseline`, `dilated_bottleneck`, or `additive_skips`.
* `--epochs`: Maximum number of training epochs (default: `10`, fast CPU default: `5`).
* `--batch_size`: Batch size (default: `16`).
* `--base_channels`: Channel width of initial encoder stage (default: `16`).
* `--resolution`: Height and width of input tensors (default: `96 128`).
* `--lr`: Peak learning rate for AdamW (default: `1e-3`).
* `--seed`: Random seed for strict reproducibility (default: `42`).
* `--patience`: Early stopping patience on validation mIoU (default: `4`).

Checkpoints and logs are saved to `experiments/<experiment_name>/`.

---

### 3. Evaluation on Test Set
Evaluate any saved checkpoint on the held-out test split (233 images):

```bash
# Evaluate Baseline
python evaluate.py --checkpoint experiments/baseline_seed42_res96x128_bc16/best_model.pth

# Evaluate Dilated Bottleneck Variant
python evaluate.py --checkpoint experiments/dilated_bottleneck_seed42_res96x128_bc16/best_model.pth
```

This generates a formatted table of:
* Overall Mean IoU (mIoU)
* Overall Pixel Accuracy
* Per-Class IoU for all 11 classes
* Trainable Parameter Count
* Saves output to `test_eval_results.json` inside the checkpoint directory.

---

### 4. Latency Benchmarking
Latency measurement is built directly into `evaluate.py`. It benchmarks single-frame CPU execution under controlled conditions:
* **Batch Size:** `1` (standard for real-time edge/CPU inference)
* **Warm-up Iterations:** `30` (stabilizes CPU caches and branch predictors)
* **Timed Iterations:** `100` (high-resolution monotonic timer `time.perf_counter`)
* **Metrics:** Mean latency (ms), standard deviation (ms), min/max latency, and throughput (Frames Per Second).

---

### 5. Visualizations & Failure Case Analysis
Generate side-by-side visual panels and error maps:

```bash
python visualize.py \
  --baseline experiments/baseline_seed42_res96x128_bc16/best_model.pth \
  --variant experiments/dilated_bottleneck_seed42_res96x128_bc16/best_model.pth \
  --output_dir visualizations
```

#### Output Artifacts:
1. `visualizations/comparative_predictions.png`: Shows Input Image, Ground Truth, Baseline Prediction, and Dilated Variant Prediction.
2. `visualizations/failure_case_analysis.png`: Highlights the highest-error samples with a dedicated binary Error Map (Red = Misclassified, Gray = Correct).

---

## Data Contract Specifications

| Class ID | Semantic Class | Color (RGB) | Active Target | Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `Sky` | `[128, 128, 128]` | Yes | Overhead sky, clouds |
| **1** | `Building` | `[128, 0, 0]` | Yes | Buildings, walls, architectural structures |
| **2** | `Pole` | `[192, 192, 128]` | Yes | Vertical poles, street lights, sign posts |
| **3** | `Road` | `[128, 64, 128]` | Yes | Drivable road surface, asphalt lanes |
| **4** | `Pavement` | `[0, 0, 192]` | Yes | Sidewalks, pedestrian walkways, curbs |
| **5** | `Tree` | `[128, 128, 0]` | Yes | Trees, bushes, foliage, vegetation |
| **6** | `SignSymbol` | `[192, 128, 128]` | Yes | Traffic signs, signboards, signals |
| **7** | `Fence` | `[64, 64, 128]` | Yes | Fences, guardrails, barriers |
| **8** | `Car` | `[64, 0, 128]` | Yes | Cars, trucks, vans, buses |
| **9** | `Pedestrian` | `[64, 64, 0]` | Yes | Walking humans, pedestrians |
| **10** | `Bicyclist` | `[0, 128, 192]` | Yes | Cyclists and their bicycles |
| **255** *(Raw: 11)* | `Void / Ignored` | `[0, 0, 0]` | **Ignored** | Boundary ambiguity & unlabeled pixels |

* **Ignored Label Rule:** Mask pixels with raw value `11` are remapped to `255`. They do not contribute to CrossEntropyLoss and are excluded from IoU calculations.

---

## Architectures & Controlled Change Mechanism

```
Input [3, 96, 128]
  │
  ├──► Stage 1 Encoder (Conv 3->16) ─────────────┐ (Skip 1)
  │      ▼ MaxPool                               │
  ├──► Stage 2 Encoder (Conv 16->32) ──────────┐ │ (Skip 2)
  │      ▼ MaxPool                             │ │
  ├──► Stage 3 Encoder (Conv 32->64) ────────┐ │ │ (Skip 3)
  │      ▼ MaxPool                           │ │ │
  └──► Stage 4 Encoder (Conv 64->128)        │ │ │
         ▼ MaxPool                           │ │ │
   ╔═════════════════════════════════════╗   │ │ │
   ║           BOTTLENECK                ║   │ │ │
   ║ Baseline: Conv 128->128 (d=1)       ║   │ │ │
   ║ Variant:  Conv 128->128 (d=2, d=4)  ║   │ │ │
   ║ Parameters: STRICTLY 0 DELTA        ║   │ │ │
   ╚═════════════════════════════════════╝   │ │ │
         ▼                                   │ │ │
       Stage 4 Decoder (Upsample + Skip 3) ◄─┘ │ │
         ▼                                     │ │
       Stage 3 Decoder (Upsample + Skip 2) ◄───┘ │
         ▼                                       │
       Stage 2 Decoder (Upsample + Skip 1) ◄─────┘
         ▼
       Classification Head (1x1 Conv 16->11)
         ▼
Output Logits [11, 96, 128]
```

* **Exact Parameter Match:**
  * Baseline Parameters: **541,003**
  * Dilated Variant Parameters: **541,003**
  * **Parameter Delta:** **0 (0.00%)**

---

## Summary of Empirical Results

### Quantitative Comparison on Test Split (233 images)

| Metric / Attribute | Baseline (Standard $d=1$) | Controlled Variant (Dilated $d=2, 4$) | Delta (Variant $-$ Baseline) |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | **541,003** | **541,003** | **0 (0.00%)** |
| **Test Mean IoU (mIoU)** | **30.60%** | **29.49%** | **-1.11%** |
| **Test Pixel Accuracy** | **76.00%** | **74.40%** | **-1.59%** |
| **CPU Latency (mean)** | **63.29 ms** | **71.88 ms** | **+8.59 ms** |
| **Throughput (FPS)** | **15.8 FPS** | **13.9 FPS** | **-1.9 FPS** |

### Per-Class Test IoU Comparison

| Class ID | Semantic Class | Baseline IoU (%) | Dilated Variant IoU (%) | Delta (%) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | `Sky` | 87.62% | 87.15% | -0.48% |
| **1** | `Building` | 66.37% | 62.26% | -4.11% |
| **2** | `Pole` | 0.73% | 0.97% | **+0.24%** |
| **3** | `Road` | 76.46% | 74.59% | -1.87% |
| **4** | `Pavement` | 20.82% | 20.02% | -0.80% |
| **5** | `Tree` | 44.01% | 36.57% | -7.44% |
| **6** | `SignSymbol` | 0.00% | 0.00% | 0.00% |
| **7** | `Fence` | 0.00% | 0.00% | 0.00% |
| **8** | `Car` | 40.60% | 42.81% | **+2.20%** |
| **9** | `Pedestrian` | 0.00% | 0.00% | 0.00% |
| **10** | `Bicyclist` | 0.00% | 0.00% | 0.00% |

---

## What the Experiment Does NOT Establish

1. **Does not establish general inferiority of dilation:** The slight drop in aggregate mIoU is specific to isolated bottleneck dilation without multi-scale Atrous Spatial Pyramid Pooling (ASPP). In architectures with parallel multi-rate branches (DeepLabV3+), atrous convolution prevents the gridding artifact observed on high-frequency classes (`Tree`, `Building`).
2. **Does not establish rare class behavior:** `Pedestrian`, `Bicyclist`, and `Fence` achieved 0% IoU because of extreme class imbalance (>75% pixels belong to `Sky`, `Road`, `Building`). This experiment does not test class-weighted loss (e.g. median frequency weighting or focal loss).
3. **Does not characterize native sub-pixel boundary detail:** Training was performed at $96 \times 128$ for CPU speed; native resolution ($360 \times 480$ / $720 \times 960$) boundary refinement behavior is unobserved.
4. **Does not establish out-of-domain generalization:** Results reflect daylight UK driving conditions (CamVid) and do not establish transferability to night, snow, rain, or indoor domains.

---

## Submission & Citations

### References
* **CamVid Database:** Brostow, G. J., Shotton, J., Fauqueur, J., & Cipolla, R. *"Semantic object classes in video: A high-definition ground truth database"*, Pattern Recognition Letters, 30(2), 88–97, 2009.
* **SegNet Architecture:** Badrinarayanan, V., Kendall, A., & Cipolla, R. *"SegNet: A deep convolutional encoder-decoder architecture for image segmentation"*, IEEE TPAMI, 39(12), 2481–2495, 2017.
* **Dilated Convolutions & Gridding:** Yu, F., & Koltun, V. *"Multi-scale context aggregation by dilated convolutions"*, ICLR, 2016; Wang et al., *"Understanding Convolution for Semantic Segmentation"*, WACV, 2018.

### Course Attribution
* **Course:** Advanced Computer Vision (CS107203CS) | Unit 3
* **Submission Date:** November 2026
