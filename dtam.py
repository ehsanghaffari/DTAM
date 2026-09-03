"""Directional-Texture Attention Module (DTAM).

This file contains only the DTAM feature-recalibration block used in the
pavement crack-segmentation study. It has no dataset, training, evaluation,
or result dependencies.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        padding=0,
        dilation=1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DirectionalTextureAttentionModule(nn.Module):
    """Directional-Texture Attention Module (DTAM).

    The module recalibrates a feature tensor using three parallel branches:

    * height-axis branch: ``kernel_size x 1``
    * width-axis branch: ``1 x kernel_size``
    * texture branch: ``3 x 3`` with configurable dilation

    The branch outputs are concatenated and fused to produce a sigmoid
    attention map ``A``. The default residual multiplicative gate is

        y = x * (1 + A)

    Parameters
    ----------
    channels:
        Number of input/output feature channels.
    kernel_size:
        Directional kernel length. The study default is 15.
    dilation:
        Dilation of the 3 x 3 texture branch. The study default is 2.
    reduction:
        Channel-reduction ratio for the branch projections. The study default
        is 4.
    min_channels:
        Minimum number of projected branch channels. The implementation used
        in the study uses 16.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dilation: int = 2,
        reduction: int = 4,
        min_channels: int = 16,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden = max(channels // reduction, min_channels)
        directional_pad = kernel_size // 2

        self.height_branch = nn.Sequential(
            ConvBNReLU(
                channels,
                hidden,
                kernel_size=(kernel_size, 1),
                padding=(directional_pad, 0),
            ),
            ConvBNReLU(hidden, hidden, kernel_size=1),
        )

        self.width_branch = nn.Sequential(
            ConvBNReLU(
                channels,
                hidden,
                kernel_size=(1, kernel_size),
                padding=(0, directional_pad),
            ),
            ConvBNReLU(hidden, hidden, kernel_size=1),
        )

        self.texture_branch = nn.Sequential(
            ConvBNReLU(
                channels,
                hidden,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            ConvBNReLU(hidden, hidden, kernel_size=1),
        )

        self.fusion = nn.Sequential(
            ConvBNReLU(hidden * 3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        """Apply DTAM to ``x``.

        Set ``return_attention=True`` to return ``(output, attention)``.
        """

        h = self.height_branch(x)
        w = self.width_branch(x)
        t = self.texture_branch(x)

        attention = self.fusion(torch.cat((h, w, t), dim=1))
        output = x * (1.0 + attention)

        if return_attention:
            return output, attention
        return output


# Short alias for convenient use.
DTAM = DirectionalTextureAttentionModule


__all__ = ["DTAM", "DirectionalTextureAttentionModule"]
