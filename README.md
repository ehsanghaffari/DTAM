# DTAM: A Directional Topology Attention Module for Pavement Crack Segmentation in 3D Line-Scan Inspection Imagery

[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)

Official implementation of:

> **DTAM: A Directional Topology Attention Module for Pavement Crack Segmentation in 3D Line-Scan Inspection Imagery**  
> Ehsan Ghaffari¹, Kelvin C.P. Wang¹, Neda Nazemi², Phil Barutha¹  
> ¹ Department of Civil Engineering, Montana State University, Bozeman, MT, USA  
> ² Department of Computer Science, Montana State University, Bozeman, MT, USA  
> Corresponding author: Ehsan Ghaffari  
> Email: ehsanghaffari@montana.edu

**ORCID**
- Ehsan Ghaffari: https://orcid.org/0009-0002-6053-4269
- Kelvin C.P. Wang: https://orcid.org/0000-0003-1346-3402
- Neda Nazemi: https://orcid.org/0000-0003-0903-3223
- Phil Barutha: https://orcid.org/0000-0003-0897-1533

---

## Overview

Pavement crack segmentation from 3D line-scan imagery is challenging due to three geometric properties of cracks: extreme elongation, severe class imbalance (< 5% foreground pixels), and topologically complex branching patterns.

We propose **DTAM (Directional Topology Attention Module)**, a plug-in attention block that replaces standard skip-connection transfer in U-Net decoders with a topology-sensitive, direction-aware feature recalibration step. DTAM uses three parallel branches:

- **Longitudinal branch** — 15×1 convolution for vertically oriented cracks
- **Transverse branch** — 1×15 convolution for horizontally oriented cracks
- **Texture branch** — 3×3 dilated convolution (dilation=2) for fine-scale crack boundaries

The three branch outputs are fused through a learned gated sigmoid and applied as a residual attention map:

$$\hat{x} = x \odot (1 + A)$$

This residual formulation preserves gradient flow and reduces to identity when attention responses are small, improving optimization stability.

DTAM is **architecture-agnostic** and can be inserted into any skip-connection-based decoder with minimal overhead (~1.8M additional parameters, +2.6% over EfficientNet-B7 U-Net baseline).

---

## Results

Evaluated on a held-out test set of **813 images** (never seen during training or validation):

| Model | F1 (%) | IoU (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|
| Baseline U-Net + EB7 | — | — | — | — |
| SegNet + EB7 | — | — | — | — |
| DeepLabV3+ + EB7 | — | — | — | — |
| Attention U-Net + EB7 | — | — | — | — |
| U-Net++ + EB7 | — | — | — | — |
| FPN + EB7 | — | — | — | — |
| **DTAM + U-Net + EB7 (ours)** | **93.56** | **87.04** | — | **93.42** |

> Full quantitative results and qualitative comparisons are reported in the paper.

---

## Architecture

```
Input (512×1024×3)
       │
  ┌────▼────────────────────────────────┐
  │      EfficientNet-B7 Encoder        │
  │   Stage 1→2→3→4→5 (5 skip feats)   │
  └────┬────────────────────────────────┘
       │ bottleneck (32×64)
  ┌────▼────────────────────────────────┐
  │         U-Net Decoder               │
  │  At each skip connection:           │
  │  encoder feat → [DTAM] → decoder   │
  └────┬────────────────────────────────┘
       │
  ┌────▼────┐
  │ Sigmoid │ → Binary crack mask (512×1024)
  └─────────┘
```

**DTAM internal structure:**
```
skip feature x (C×H×W)
       ├──→ [15×1 conv → 1×1 conv] → longitudinal branch
       ├──→ [1×15 conv → 1×1 conv] → transverse branch
       └──→ [3×3 dilated conv → 1×1 conv] → texture branch
                          │
                   concat (3C/4 × H × W)
                          │
                   [1×1 Conv-BN-ReLU → 1×1 Conv → Sigmoid] → A
                          │
                   x̂ = x ⊙ (1 + A)
```

---

## Installation

```bash
git clone https://github.com/ehsanghaffari/DTAM.git
cd DTAM
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU (tested on NVIDIA RTX 3090, 24 GB VRAM)

---

## Dataset

The dataset used in this study consists of **4,414 matched image–mask pairs** acquired with a vehicle-mounted 3D line-scan pavement inspection system under controlled LED illumination at a fixed nadir viewing angle.

| Split | Images |
|---|---|
| Training | 3,241 |
| Validation | 360 |
| Held-out Test | 813 |
| **Total** | **4,414** |

- Image format: BMP (greyscale intensity, replicated to 3 channels)
- Resolution: 512 × 1024 pixels
- Masks: binary (crack = 255, background = 0)

> **Note:** The dataset is proprietary and cannot be shared publicly. Please refer to the paper for full acquisition details.

---

## Usage

### Training

1. Set your dataset paths in `Train.py`:

```python
IMAGE_DIR  = "data/images"   # path to your image folder
MASK_DIR   = "data/masks"    # path to your mask folder
OUTPUT_DIR = "outputs/DTAM_BCE_Dice"
```

2. Run training:

```bash
python Train.py
```

**Outputs saved to `OUTPUT_DIR`:**
- `best_model.pth` — checkpoint at peak validation F1
- `last_model.pth` — checkpoint at final epoch
- `training_log.csv` / `training_log.xlsx` — per-epoch metrics
- `training_summary.txt` — final summary

**Training configuration (as used in the paper):**

| Parameter | Value |
|---|---|
| Encoder | EfficientNet-B7 (ImageNet pretrained) |
| Input resolution | 512 × 1024 |
| Batch size | 2 |
| Optimizer | Adam (lr=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |
| Epochs | 200 |
| Loss | BCE + Dice |
| Mixed precision | FP16 (torch.amp) |
| Random seed | 42 |

---

### Inference

1. Set your paths in `Inference_DTAM.py`:

```python
INPUT_DIR  = "data/test_images"
MODEL_PATH = "outputs/DTAM_BCE_Dice/best_model.pth"
OUTPUT_DIR = "outputs/DTAM_BCE_Dice/results"
```

2. Run inference:

```bash
python Inference_DTAM.py
```

**Outputs saved per image:**
- `masks/` — binary crack segmentation masks
- `overlays/` — red crack overlay on original image
- `probability_maps/` — raw sigmoid probability maps
- `inference_log.csv` — per-image inference time

---

## Training Hyperparameter Details

### Augmentation Policy

| Augmentation | Parameters | Probability |
|---|---|---|
| Resize | H=512, W=1024 | Always |
| Horizontal Flip | — | 0.5 |
| Vertical Flip | — | 0.2 |
| Rotate | limit=±8° | 0.5 |
| RandomBrightnessContrast | ±0.12 | 0.4 |
| GaussNoise | — | 0.2 |
| Normalize | ImageNet stats | Always |

---

## Repository Structure

```
DTAM/
├── README.md
├── requirements.txt
├── Train.py                  # Training script
├── Inference_DTAM.py       # Inference script
├── data/
│   └── README.md             # Dataset description
└── outputs/                  # Generated outputs (gitignored)
    └── .gitkeep
```

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@misc{ghaffari2026dtam,
  title        = {DTAM: A Directional Topology Attention Module for Pavement Crack Segmentation in 3D Line-Scan Inspection Imagery},
  author       = {Ghaffari, Ehsan and Wang, Kelvin C. P. and Nazemi, Neda and Barutha, Phil},
  year         = {2026},
  note         = {Manuscript in preparation},
  url          = {https://github.com/ehsanghaffari/DTAM}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii
- [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation
- EfficientNet-B7 encoder pretrained on ImageNet via `timm`
