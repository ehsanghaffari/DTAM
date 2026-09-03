# DTAM

**Directional-Texture Attention Module (DTAM)** for pavement crack segmentation in 3D line-scan imagery.

This repository contains only the DTAM implementation used in the study. It does not include datasets, trained weights, evaluation outputs, or reported results.

## Module

DTAM recalibrates a feature tensor using three parallel branches:

- `15 x 1` height-axis branch,
- `1 x 15` width-axis branch,
- dilated `3 x 3` texture branch with dilation `2`.

The three responses are concatenated and fused into a sigmoid attention map `A`. The default gate is:

```text
y = x * (1 + A)
```

The study configuration uses reduction ratio `r = 4` with a minimum projected width of 16 channels.

## Usage

```python
import torch
from dtam import DTAM

x = torch.randn(2, 64, 256, 512)
module = DTAM(channels=64)
y = module(x)

print(y.shape)  # torch.Size([2, 64, 256, 512])
```

To also obtain the learned attention map:

```python
y, attention = module(x, return_attention=True)
```

## Requirements

```bash
pip install -r requirements.txt
```

## File

- `dtam.py` — Directional-Texture Attention Module implementation.
