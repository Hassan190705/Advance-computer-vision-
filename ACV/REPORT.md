# Assignment B: Reproducible Segmentation Experiment Report

**Name**: Hassan Ashraf
**Roll No**: 23115040
**Subject:** Advanced Computer Vision
**Experiment Objective:** Build a fully reproducible semantic segmentation experiment evaluating the impact of a single architectural mechanism on an annotated public dataset under an explicit compute and parameter budget.

---

## 1. Task 1: Data Contract

### 1.1 Dataset Definition and Citation
* **Dataset:** Cambridge-driving Labeled Video Database (**CamVid** - 11-Class Academic Benchmark).
* **Reference:** Brostow, G. J., Shotton, J., Fauqueur, J., & Cipolla, R. *"Semantic object classes in video: A high-definition ground truth database"*, Pattern Recognition Letters, 2009.
* **Source Repository:** [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
* **Ground Truth Nature:** Strictly verified, human-annotated ground truth (no algorithm-generated pseudo-ground truth).

### 1.2 Splits and Counts
The dataset follows the official benchmark partition:
* **Train Split:** 367 images & annotated masks
* **Validation Split:** 101 images & annotated masks
* **Test Split:** 233 images & annotated masks
* **Total Image Bank:** 701 annotated frames ($360 \times 480$ native resolution)

### 1.3 Class IDs, Color Map, and Ignored Labels
The annotation contract specifies **11 active semantic classes** and an explicit **unlabeled/void label** excluded from loss gradients and evaluation metrics:

| Class ID | Semantic Class | Visual Hex / RGB | Active Target? | Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `Sky` | `[128, 128, 128]` | Yes | Overhead clear, overcast, or cloudy sky |
| **1** | `Building` | `[128, 0, 0]` | Yes | Buildings, houses, walls, architectural structures |
| **2** | `Pole` | `[192, 192, 128]` | Yes | Traffic sign poles, lamp posts, vertical thin poles |
| **3** | `Road` | `[128, 64, 128]` | Yes | Drivable road surface, asphalt, street lanes |
| **4** | `Pavement` | `[0, 0, 192]` | Yes | Sidewalks, pedestrian pathways, curbs |
| **5** | `Tree` | `[128, 128, 0]` | Yes | Foliage, trees, bushes, vegetation |
| **6** | `SignSymbol` | `[192, 128, 128]` | Yes | Traffic signs, signals, billboard placards |
| **7** | `Fence` | `[64, 64, 128]` | Yes | Guardrails, wire fencing, barriers |
| **8** | `Car` | `[64, 0, 128]` | Yes | Sedans, SUVs, trucks, buses, motorized vans |
| **9** | `Pedestrian` | `[64, 64, 0]` | Yes | Walking humans, pedestrians, workers |
| **10** | `Bicyclist` | `[0, 128, 192]` | Yes | Cyclists and their bicycles |
| **255** *(Raw: 11)* | `Void / Ignored` | `[0, 0, 0]` | **Ignored** | Ambiguous boundary pixels & unlabeled regions |

> **Contract Rule on Ignored Label:** Raw mask pixel value `11` is remapped to `ignore_index = 255` during tensor conversion. Cross-entropy loss ignores all pixels labeled `255`, and metric computation excludes them from both numerator and denominator in per-class IoU.

### 1.4 Image and Mask Transforms
* **Spatial Resizing:** Downsampled to $(96, 128)$ preserving the native $3:4$ aspect ratio. Image interpolation uses **Bilinear**; mask interpolation uses **Nearest Neighbor** to preserve discrete class integer IDs without interpolation artifacts.
* **Data Augmentation (Train only):** Random Horizontal Flip ($p=0.5$).
* **Normalization:** Standard channel-wise z-score normalization using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 2. Task 2: Baseline Specification

| Component | Specification |
| :--- | :--- |
| **Architecture** | Modest 4-Stage Encoder–Decoder (Lightweight U-Net) with Concatenation Skip Connections |
| **Channel Progression** | Encoder: $3 \to 16 \to 32 \to 64 \to 128$; Decoder: $128 \to 64 \to 32 \to 16 \to 11$ |
| **Bottleneck Layers** | Two $3\times 3$ standard convolutions ($d=1$) with BatchNorm and ReLU |
| **Total Parameters** | **541,003** trainable parameters |
| **Random Seed** | Fixed to **`42`** (`torch`, `numpy`, `random`, `torch.backends.cudnn.deterministic=True`) |
| **Input Resolution** | $96 \times 128$ ($H \times W$, divisible by $16$ across all 4 downsampling stages) |
| **Batch Size** | 16 |
| **Optimizer** | **AdamW** ($\text{learning rate} = 10^{-3}$, $\text{weight decay} = 10^{-4}$, $\beta_1=0.9, \beta_2=0.999$) |
| **LR Scheduler** | **CosineAnnealingLR** ($T_{\max}=5, \eta_{\min}=10^{-5}$) |
| **Loss Function** | CrossEntropyLoss with `ignore_index = 255` |
| **Stopping Rule** | Fixed 5 epochs with early stopping rule (patience = 4 on validation mIoU) |

---

## 3. Task 3: Controlled Change Mechanism

### 3.1 Mechanism: Receptive Field Dilation (Atrous Bottleneck)
* **Isolated Variable:** The convolution mechanism within the lowest-resolution bottleneck ($6 \times 8$ feature map).
  * **Baseline Mechanism:** Standard $3 \times 3$ convolutions with **dilation rate $d=1$** (stride 1, padding 1).
  * **Controlled Variant:** Dilated (Atrous) $3 \times 3$ convolutions with **dilation rate $d=2$** (layer 1, padding 2) and **dilation rate $d=4$** (layer 2, padding 4).
* **Theoretical Hypothesis:** In road scene understanding, global context (e.g. associating road surfaces with horizon lines and differentiating upper sky from lower sidewalks) is critical. Dilated convolutions expand the receptive field exponentially without losing spatial resolution or requiring downsampling pooling layers.
* **Strict Parameter Budget Control:** Because atrous convolution samples weights at non-consecutive intervals without adding weights, the tensor shapes are mathematically identical:
  $$\text{Baseline Bottleneck Weights} \in \mathbb{R}^{128 \times 128 \times 3 \times 3} \quad \equiv \quad \text{Dilated Bottleneck Weights} \in \mathbb{R}^{128 \times 128 \times 3 \times 3}$$
  $$\Delta \text{ Parameters} = 0 \quad (0.00\% \text{ discrepancy})$$

---

## 4. Task 4: Empirical Evidence

### 4.1 Quantitative Results on Held-out Test Split (233 images)

| Evaluation Metric / Attribute | Baseline (Standard Conv $d=1$) | Controlled Variant (Dilated Bottleneck $d=2, 4$) | Delta (Variant $-$ Baseline) |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | **541,003** | **541,003** | **0 (0.00%)** |
| **Test Mean IoU (mIoU)** | **30.60%** | **29.49%** | **-1.11%** |
| **Test Pixel Accuracy** | **76.00%** | **74.40%** | **-1.59%** |
| **Inference Latency (CPU Mean)** | **63.29 ms** ($\pm 56.91$ ms) | **71.88 ms** ($\pm 45.84$ ms) | **+8.59 ms** |
| **Throughput (FPS)** | **15.8 FPS** | **13.9 FPS** | **-1.9 FPS** |

### 4.2 Per-Class IoU Breakdown (Official Test Split)

| Class ID | Semantic Class | Baseline IoU (%) | Dilated Variant IoU (%) | Absolute Delta (%) | Key Structural Characteristic |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | `Sky` | 87.62% | 87.15% | -0.48% | Large homogeneous upper-frame region |
| **1** | `Building` | 66.37% | 62.26% | -4.11% | Medium-large vertical structural region |
| **2** | `Pole` | 0.73% | 0.97% | **+0.24%** | Extremely thin vertical structure |
| **3** | `Road` | 76.46% | 74.59% | -1.87% | Large continuous horizontal region |
| **4** | `Pavement` | 20.82% | 20.02% | -0.80% | Road-adjacent sidewalk surface |
| **5** | `Tree` | 44.01% | 36.57% | -7.44% | High-frequency textured vegetation |
| **6** | `SignSymbol` | 0.00% | 0.00% | 0.00% | Small sparse object (insufficient samples) |
| **7** | `Fence` | 0.00% | 0.00% | 0.00% | Thin grid boundary |
| **8** | `Car` | 40.60% | 42.81% | **+2.20%** | Compact vehicle contours with varied shapes |
| **9** | `Pedestrian` | 0.00% | 0.00% | 0.00% | Small dynamic instance (class imbalance) |
| **10** | `Bicyclist` | 0.00% | 0.00% | 0.00% | Rare class (< 0.5% labeled pixels) |

### 4.3 Key Empirical Insights
1. **Vehicles Benefit from Broader Context:** The `Car` category exhibited a **+2.20% absolute IoU boost** (from 40.60% to 42.81%) with dilated bottleneck convolutions. The expanded receptive field allows the model to encompass entire vehicle silhouettes rather than misclassifying car roofs as roads or building structures.
2. **Gridding Effect on High-Frequency Textures:** Classes with fine, dispersed high-frequency patterns (`Tree`: $-7.44\%$, `Building`: $-4.11\%$) experienced slight degradation under dilation. This aligns with well-documented computer vision literature (Wang et al., *"Understanding Convolution for Semantic Segmentation"*, WACV 2018): atrous convolutions without dense multi-scale pooling can suffer from the "gridding artifact," skipping local high-frequency details.
3. **Severe Class Imbalance on Rare Categories:** Both models struggled with extremely sparse classes (`SignSymbol`, `Fence`, `Pedestrian`, `Bicyclist`), achieving 0% IoU. In CamVid, `Sky`, `Road`, and `Building` occupy $>75\%$ of all labeled pixels, dominating standard cross-entropy loss without class-frequency reweighting.

---

## 5. Qualitative Predictions and Failure Analysis

### 5.1 Comparative Predictions Across Models
The comparative visual panel demonstrates predictions for input scenes, official ground truth, baseline U-Net, and the dilated variant:

Visual output generated at: [`visualizations/comparative_predictions.png`](file:///c:/Users/sapur/ACV/visualizations/comparative_predictions.png)

### 5.2 Documented Failure Case Analysis
The dedicated failure analysis identifies spatial regions of model breakdown using an explicit error map (Red = Error, Gray = Correct):

Visual output generated at: [`visualizations/failure_case_analysis.png`](file:///c:/Users/sapur/ACV/visualizations/failure_case_analysis.png)

#### Specific Observed Failure Modes:
* **Failure Mode A: Road vs. Pavement Boundary Confusion:** Both models exhibit significant boundary bleed between `Road` (purple) and `Pavement` (blue). At downsampled resolutions, curbs lack sharp gradient transitions, leading to ambiguous boundaries.
* **Failure Mode B: Thin Vertical Structure Disappearance:** Thin `Pole` (beige) and `SignSymbol` structures are frequently absorbed into background buildings or foliage. Standard stride-2 pooling downsamples feature maps by $16\times$, reducing single-pixel poles to sub-pixel features before the bottleneck.
* **Failure Mode C: Distant and Occluded Cars:** Small parked cars distant from the ego-vehicle are misclassified as `Road` or `Building`, while nearby foreground cars are reliably segmented.

---

## 6. What the Experiment Does NOT Establish

To maintain academic rigor, the following limitations are explicitly noted:

1. **Does NOT Establish Superiority of Standard Convolutions Over Dilation in General:** While the standard baseline attained slightly higher aggregate mIoU (30.60% vs 29.49%), this is specific to a 5-epoch training budget on a lightweight architecture without multi-rate Atrous Spatial Pyramid Pooling (ASPP). Modern dilated architectures (e.g. DeepLabV3+) employ parallel dilation rates ($d=\{6, 12, 18\}$) and image pooling to prevent the gridding artifact.
2. **Does NOT Establish True Performance on Rare Classes:** Classes such as `Pedestrian` and `Bicyclist` failed to register positive IoU due to severe class imbalance. This experiment does not test whether focal loss, boundary-aware loss, or median-frequency class weighting would enable the models to segment rare classes.
3. **Does NOT Establish Ultra-High-Resolution Boundary Precision:** To ensure reproducible execution on CPU hardware, models were trained at $96 \times 128$. This experiment does not characterize full-resolution boundary refinement (e.g. at native $720 \times 960$ or $360 \times 480$), where high-frequency edge filters play a larger role.
4. **Does NOT Establish Cross-Dataset Generalization:** All training and testing was conducted on the CamVid Cambridge driving dataset. These findings do not guarantee identical transferability to night driving, adverse weather (rain/fog), or indoor segmentation datasets (e.g., NYUv2, ADE20K).

---

## 7. Submission Artifacts and Environment Versions

### 7.1 Environment Specifications
* **Operating System:** Windows 10 (Build 10.0.19045)
* **CPU Hardware:** Intel(R) Core(TM) Architecture with Integrated HD Graphics 520
* **Python Version:** `3.14.3`
* **PyTorch Version:** `2.15.0+cpu`
* **Torchvision Version:** `0.29.2+cpu`
* **NumPy Version:** `2.5.2`
* **Pandas Version:** `3.1.1`
* **Pillow Version:** `12.3.0`

### 7.2 Codebase Manifest
* **Data Contract:** [`src/data_contract.py`](file:///c:/Users/sapur/ACV/src/data_contract.py) — Defines class taxonomy, color mapping, ignore label handling, transforms, and verification logic.
* **Model Architectures:** [`src/models.py`](file:///c:/Users/sapur/ACV/src/models.py) — Implements Baseline and Dilated Bottleneck models with strictly matched parameter counts (541,003 parameters).
* **Streaming Metrics:** [`src/metrics.py`](file:///c:/Users/sapur/ACV/src/metrics.py) — Confusion matrix accumulator computing per-class IoU, mIoU, and pixel accuracy with `ignore_index = 255`.
* **Training Script:** [`train.py`](file:///c:/Users/sapur/ACV/train.py) — Reproducible training pipeline with seed 42, AdamW, Cosine Annealing, early stopping, and structured logging.
* **Evaluation Script:** [`evaluate.py`](file:///c:/Users/sapur/ACV/evaluate.py) — Exact evaluation runner generating test set metrics and latency benchmarking.
* **Visualization Script:** [`visualize.py`](file:///c:/Users/sapur/ACV/visualize.py) — Generates comparative panels and failure case analysis figures.
* **Master Pipeline Runner:** [`run_all_experiments.py`](file:///c:/Users/sapur/ACV/run_all_experiments.py) — End-to-end automation executing data verification, baseline training, variant training, evaluation, latency tests, and figure generation.
* **Training Logs:**
  * Baseline Log: [`experiments/baseline_seed42_res96x128_bc16/training.log`](file:///c:/Users/sapur/ACV/experiments/baseline_seed42_res96x128_bc16/training.log)
  * Dilated Variant Log: [`experiments/dilated_bottleneck_seed42_res96x128_bc16/training.log`](file:///c:/Users/sapur/ACV/experiments/dilated_bottleneck_seed42_res96x128_bc16/training.log)
* **Master Results JSON:** [`experiments/consolidated_results.json`](file:///c:/Users/sapur/ACV/experiments/consolidated_results.json)
