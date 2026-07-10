# Directional Topology Attention for Pavement Crack Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)

Official implementation of **Directional Topology Attention for Automated Pavement Crack Detection in 3D Line-Scan Inspection Imagery**.

**Authors:** Ehsan Ghaffari, Kelvin C. P. Wang, Neda Nazemi, and Philip Barutha  
Montana State University, Bozeman, Montana, USA  
Corresponding author: Ehsan Ghaffari (ehsanghaffari@montana.edu)

## Overview

The Directional Topology Attention Module (DTAM) is a skip-connection feature-recalibration block developed for thin, elongated, and interconnected pavement cracks in laser-derived range imagery. It contains three parallel branches:

- a `15 x 1` longitudinal branch,
- a `1 x 15` transverse branch, and
- a dilated `3 x 3` texture branch with dilation 2.

The branch responses are concatenated, fused through `1 x 1` convolutions, and converted into a sigmoid attention map `A`. The skip feature is recalibrated using residual multiplicative gating:

```text
x_tilde = x * (1 + A)
```

The residual form allows the module to remain close to an identity mapping when attention responses are small.

## Network configuration

The paper configuration uses a U-Net decoder with an ImageNet-pretrained EfficientNet-B7 encoder. A single-channel pavement range image is replicated to three channels and resized to `512 x 1024` pixels.

| Component | Configuration |
|---|---|
| Encoder | EfficientNet-B7, ImageNet pretrained |
| Encoder skip channels | 64, 48, 80, 224 |
| Bottleneck | `16 x 32 x 640` |
| Decoder channels | 256, 128, 64, 32, 16 |
| DTAM placement | Four encoder-decoder skip connections |
| Output | One-channel sigmoid crack-probability map |

DTAM adds approximately 1.8 million parameters and 1.7 GFLOPs relative to the corresponding EfficientNet-B7 U-Net baseline.

## Dataset

The complete study used 4,414 matched range-image and binary-mask pairs acquired using a vehicle-mounted 3D laser line-scan pavement inspection system.

| Split | Images |
|---|---:|
| Training | 3,241 |
| Validation | 360 |
| Held-out test | 813 |
| **Total** | **4,414** |

- Resolution: `512 x 1024` pixels
- Image format: laser-derived single-channel range imagery replicated to three channels
- Mask encoding: crack = 255, background = 0
- Binarization threshold: 127 when reading reference masks
- Prediction threshold: 0.5

The dataset is proprietary and cannot be distributed publicly.

## Held-out evaluation

All model configurations were evaluated on an independent held-out test set of 813 pavement range images using a fixed sigmoid threshold of 0.5.

| Model Configuration | IoU (%) | F1 (%) | Prec (%) | Rec (%) | Acc (%) |
|---|---:|---:|---:|---:|---:|
| **DTAM + BCE+Dice** | **87.21** | **93.17** | 92.82 | **93.52** | 99.69 |
| U-Net++ + BCE+Dice | 87.20 | 93.16 | 94.63 | 91.75 | **99.70** |
| Attention U-Net + BCE+Dice | 86.12 | 92.54 | 93.30 | 91.79 | 99.67 |
| Baseline (U-Net, no DTAM) | 86.03 | 92.49 | 94.67 | 90.41 | 99.68 |
| DTAM + BCE+Dice + 0.1×clDice | 86.11 | 92.53 | 93.46 | 91.63 | 99.67 |
| DTAM + BCE+Dice + 0.2×clDice | 82.63 | 90.49 | **95.80** | 85.73 | 99.60 |
| DTAM + BCE+Dice + 0.3×clDice | 83.88 | 91.23 | 91.22 | 91.25 | 99.61 |
| DeepLabV3+ + BCE+Dice | 82.55 | 90.44 | 92.37 | 88.58 | 99.59 |
| SegNet + BCE+Dice | 81.82 | 90.00 | 93.97 | 86.35 | 99.58 |
| FPN + BCE+Dice | 81.13 | 89.58 | 88.40 | 90.80 | 99.53 |
| DTAM + 0.5×FT + 0.5×clDice | 80.71 | 89.32 | 87.23 | 91.51 | 99.52 |

DTAM trained with BCE+Dice achieved the highest held-out F1 score, IoU, and recall among the evaluated configurations.

## Installation

```bash
git clone https://github.com/ehsanghaffari/DTAM.git
cd DTAM
pip install -r requirements.txt
```

Tested with an NVIDIA RTX 3090 GPU with 24 GB memory.

## Training

Set the image, mask, and output paths in `Train.py`, then run:

```bash
python Train.py
```

Paper training configuration:

| Parameter | Value |
|---|---|
| Input resolution | `512 x 1024` |
| Batch size | 2 |
| Epochs | 200 |
| Optimizer | Adam, learning rate `1e-4` |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 8 |
| Minimum learning rate | `1e-7` |
| Loss | BCE + Dice |
| Mixed precision | FP16 |
| Random seed | 42 |

Training augmentation consists of resizing, horizontal and vertical flips, rotation up to 8 degrees, random brightness/contrast adjustment, Gaussian noise, and ImageNet normalization.

## Inference

Set the paths in `Inference_DTAM.py`, then run:

```bash
python Inference_DTAM.py
```

The inference script produces binary masks, overlays, probability maps, and an inference log.

## Citation

```bibtex
@article{ghaffari2026dtam,
  title   = {Directional Topology Attention for Automated Pavement Crack Detection in 3D Line-Scan Inspection Imagery},
  author  = {Ghaffari, Ehsan and Wang, Kelvin C. P. and Nazemi, Neda and Barutha, Philip},
  year    = {2026},
  note    = {Manuscript submitted for publication},
  url     = {https://github.com/ehsanghaffari/DTAM}
}
```

## Data and code availability

The source code for the DTAM module and training pipeline is publicly available in this repository. The underlying pavement dataset was provided by Waylink Systems Corporation and is not publicly available because of confidentiality restrictions. Derived metrics and training logs supporting the reported results may be obtained from the corresponding author upon reasonable request.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This implementation uses [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), [Albumentations](https://github.com/albumentations-team/albumentations), and an EfficientNet-B7 encoder pretrained on ImageNet.