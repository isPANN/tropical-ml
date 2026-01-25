"""
Vision Models using MMP Neural Networks.

Provides convolutional architectures with MMP classifier heads for
image classification tasks (CIFAR-10, ImageNet, etc.)

Architecture pattern:
    Conv Feature Extractor → MMP Classifier Head

The convolutional backbone extracts spatial features, while the
MMP head provides tropical nonlinearity for classification.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MaxPlusLayer, MinPlusLayer
from .blocks import MMPBlock, TropicalMLP


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv → BatchNorm → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class MMPConvClassifier(nn.Module):
    """
    Convolutional network with MMP classifier head.

    Architecture:
        Conv layers (feature extraction) → Global Pool → MMP classifier

    This hybrid approach uses standard convolutions for spatial feature
    extraction and MMP layers for the classification head.

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        base_channels: Base number of channels. Default: 64.
        num_blocks: Number of conv blocks per stage. Default: 2.
        mmp_hidden: Hidden dimensions for MMP classifier. Default: [512, 256].
        dropout: Dropout rate. Default: 0.0.

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        num_blocks: int = 2,
        mmp_hidden: List[int] = [512, 256],
        dropout: float = 0.0,
        use_mmp: bool = True,
    ):
        super().__init__()

        self.use_mmp = use_mmp

        # Convolutional feature extractor
        # Stage 1: 32x32 -> 32x32
        self.stage1 = self._make_stage(in_channels, base_channels, num_blocks, stride=1)

        # Stage 2: 32x32 -> 16x16
        self.stage2 = self._make_stage(base_channels, base_channels * 2, num_blocks, stride=2)

        # Stage 3: 16x16 -> 8x8
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks, stride=2)

        # Stage 4: 8x8 -> 4x4
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, num_blocks, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        feature_dim = base_channels * 8

        if use_mmp:
            # MMP classifier
            layers = []
            dims = [feature_dim] + mmp_hidden

            for i in range(len(dims) - 1):
                layers.append(MMPBlock(dims[i], out_features=dims[i + 1], dropout=dropout))

            layers.append(nn.Linear(dims[-1], num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            # Standard MLP classifier for comparison
            layers = []
            dims = [feature_dim] + mmp_hidden + [num_classes]

            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU(inplace=True))
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))

            self.classifier = nn.Sequential(*layers)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage with conv blocks."""
        layers = []

        # First block with potential downsampling
        layers.append(ConvBlock(in_channels, out_channels, stride=stride))

        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


class MMPResNet(nn.Module):
    """
    ResNet-style architecture with MMP classifier.

    Uses residual blocks for feature extraction and MMP layers
    for the classification head.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        layers: Number of residual blocks per stage. Default: [2, 2, 2, 2].
        base_channels: Base number of channels. Default: 64.
        mmp_hidden: Hidden dimensions for MMP classifier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        layers: List[int] = [2, 2, 2, 2],
        base_channels: int = 64,
        mmp_hidden: List[int] = [512, 256],
        dropout: float = 0.0,
        use_mmp: bool = True,
    ):
        super().__init__()

        self.use_mmp = use_mmp

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0])
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        feature_dim = base_channels * 8

        if use_mmp:
            dims = [feature_dim] + mmp_hidden
            layers_list = []

            for i in range(len(dims) - 1):
                layers_list.append(MMPBlock(dims[i], out_features=dims[i + 1], dropout=dropout))

            layers_list.append(nn.Linear(dims[-1], num_classes))
            self.classifier = nn.Sequential(*layers_list)
        else:
            dims = [feature_dim] + mmp_hidden + [num_classes]
            layers_list = []

            for i in range(len(dims) - 1):
                layers_list.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers_list.append(nn.ReLU(inplace=True))
                    if dropout > 0:
                        layers_list.append(nn.Dropout(dropout))

            self.classifier = nn.Sequential(*layers_list)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual stage."""
        layers = []

        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class CIFAR10MMP(nn.Module):
    """
    MMP model optimized for CIFAR-10.

    A compact architecture suitable for 32x32 images.

    Args:
        num_classes: Number of classes. Default: 10.
        use_mmp: Use MMP classifier (True) or ReLU baseline (False).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_mmp: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 32x32
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x4
        )

        # Classifier
        feature_dim = 256 * 4 * 4

        if use_mmp:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                MaxPlusLayer(512, 512),
                MinPlusLayer(512, 256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                MaxPlusLayer(128, 128),
                MinPlusLayer(128, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ImageNetMMP(nn.Module):
    """
    MMP model for ImageNet-scale datasets.

    Uses a ResNet-style backbone with MMP classifier head.

    Args:
        num_classes: Number of classes. Default: 1000.
        use_mmp: Use MMP classifier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        use_mmp: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.use_mmp = use_mmp

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ResNet-34 style stages
        self.stage1 = self._make_stage(64, 64, 3)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        if use_mmp:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                MaxPlusLayer(1024, 1024),
                MinPlusLayer(1024, 512),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(1024, num_classes),
            )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []

        # Downsample
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels

        # Blocks
        for _ in range(num_blocks):
            layers.append(self._make_block(out_channels))

        return nn.Sequential(*layers)

    def _make_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cifar10_model(model_type: str = "mmp", **kwargs) -> nn.Module:
    """Factory function for CIFAR-10 models."""
    if model_type == "mmp":
        return CIFAR10MMP(use_mmp=True, **kwargs)
    elif model_type == "baseline":
        return CIFAR10MMP(use_mmp=False, **kwargs)
    elif model_type == "resnet_mmp":
        return MMPResNet(num_classes=10, use_mmp=True, **kwargs)
    elif model_type == "resnet_baseline":
        return MMPResNet(num_classes=10, use_mmp=False, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_imagenet_model(model_type: str = "mmp", **kwargs) -> nn.Module:
    """Factory function for ImageNet models."""
    if model_type == "mmp":
        return ImageNetMMP(use_mmp=True, **kwargs)
    elif model_type == "baseline":
        return ImageNetMMP(use_mmp=False, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "ConvBlock",
    "ResBlock",
    "MMPConvClassifier",
    "MMPResNet",
    "CIFAR10MMP",
    "ImageNetMMP",
    "create_cifar10_model",
    "create_imagenet_model",
]
