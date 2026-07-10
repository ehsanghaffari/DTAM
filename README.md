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

The dataset is proprietary and cannot be distributed publicly. Derived evaluation files are provided in [`results/`](results/).

## Updated held-out evaluation

The repository now includes a new per-image topology-aware re-evaluation of all seven architectural configurations on the 813-image held-out set. Pixel and topology metrics in the table below are per-image macro averages. These values should not be mixed with dataset-level micro-averaged metrics reported by earlier evaluation scripts.

| Model | F1 (%) | IoU (%) | Precision (%) | Recall (%) | clDice (%) | Topology precision (%) | Topology sensitivity (%) | Fragmentation ratio | Skeleton-length error (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention U-Net | 75.49 | 64.00 | 68.25 | 87.23 | 83.49 | 81.58 | 87.74 | 0.82 | 29.45 |
| U-Net++ | 75.28 | 63.86 | 68.72 | 86.44 | 83.14 | 81.51 | 87.30 | 0.82 | 31.58 |
| Baseline U-Net | 75.13 | 63.83 | 69.60 | 85.69 | **83.80** | **83.95** | 86.66 | 0.79 | 51.41 |
| **DTAM** | 74.38 | 62.72 | 66.20 | **88.43** | 81.19 | 76.37 | **89.88** | **1.00** | 60.53 |
| DeepLabV3+ | 72.98 | 61.20 | 67.12 | 83.05 | 81.51 | 81.67 | 83.69 | 0.77 | **25.39** |
| FPN | 72.62 | 60.53 | 64.65 | 86.08 | 81.91 | 79.32 | 87.46 | 0.92 | 47.00 |
| SegNet | 62.53 | 50.10 | **79.36** | 56.72 | 69.59 | 89.28 | 60.36 | 1.26 | 39.79 |

In this updated macro-averaged evaluation, DTAM provides the highest recall and topology sensitivity and a fragmentation ratio closest to 1.0. Attention U-Net has the highest macro F1 and IoU. This distinction is retained explicitly so the repository reports the new results transparently rather than presenting DTAM as uniformly best across every metric.

The complete model-level summaries and the 813 per-image records for every architecture are available in [`results/`](results/).

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

## Evaluation files

```text
results/
├── all_models_summary_metrics.csv
├── attention_unet_summary_metrics.csv
├── attention_unet_per_image_metrics.csv
├── baseline_unet_summary_metrics.csv
├── baseline_unet_per_image_metrics.csv
├── deeplabv3plus_summary_metrics.csv
├── deeplabv3plus_per_image_metrics.csv
├── dtam_summary_metrics.csv
├── dtam_per_image_metrics.csv
├── fpn_summary_metrics.csv
├── fpn_per_image_metrics.csv
├── segnet_summary_metrics.csv
├── segnet_per_image_metrics.csv
├── unetpp_summary_metrics.csv
└── unetpp_per_image_metrics.csv
```

The summary files contain F1, IoU, precision, recall, accuracy, clDice, topology precision, topology sensitivity, connected-component error, fragmentation ratio, fragmentation-ratio error, and skeleton-length error. The per-image files preserve the image-level values used to calculate the macro averages.

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