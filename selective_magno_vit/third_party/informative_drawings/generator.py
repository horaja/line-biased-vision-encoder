"""
Minimal Generator architecture vendored from the informative-drawings repository
by carolineec (https://github.com/carolineec/informative-drawings).
Only the inference-time components required to run pretrained checkpoints are kept.
"""

from typing import Optional

import torch
import torch.nn as nn

# The original code defaults to instance normalization.
_NORM = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    """Simple residual block used by the Generator."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            _NORM(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            _NORM(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    ResNet-based image-to-image generator.

    Args:
        input_nc: Number of input channels (e.g., 3 for RGB)
        output_nc: Number of output channels (e.g., 1 for grayscale line drawing)
        n_residual_blocks: Count of residual blocks to use (default matches upstream: 9)
        sigmoid: Whether to apply sigmoid at the output (upstream default is True)
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        n_residual_blocks: int = 9,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()

        # Initial convolution block
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            _NORM(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling (two strided convolutions)
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                _NORM(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        # Residual blocks
        self.model2 = nn.Sequential(*[ResidualBlock(in_features) for _ in range(n_residual_blocks)])

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                _NORM(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7)]
        if sigmoid:
            model4.append(nn.Sigmoid())
        self.model4 = nn.Sequential(*model4)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass. `cond` is unused but kept for compatibility with upstream signatures.
        """
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out
