"""
Vision Models with Tropical Classifiers.

Conv backbone for feature extraction + tropical classifier head.

Architecture:
    Conv Feature Extractor → Global Pool → Linear → MaxPlus → MinPlus → Linear
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MaxPlusAffine, MinPlusAffine
from .blocks import TropicalBlock


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class CIFAR10Tropical(nn.Module):
    """
    Tropical model for CIFAR-10.

    Conv backbone + tropical classifier head.

    Args:
        num_classes: Number of classes.
        use_tropical: Use tropical classifier (True) or ReLU baseline (False).
        dropout: Dropout rate.
    """

    def __init__(self, num_classes: int = 10, use_tropical: bool = True, dropout: float = 0.1):
        super().__init__()

        # Conv feature extractor
        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2: 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3: 8x8 → 4x4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        feature_dim = 256 * 4 * 4

        if use_tropical:
            # Tropical classifier: Linear → MaxPlus → MinPlus → Linear
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                MaxPlusAffine(512),
                MinPlusAffine(512),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                MaxPlusAffine(128),
                MinPlusAffine(128),
                nn.Linear(128, num_classes),
            )
        else:
            # ReLU baseline
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Alias
CIFAR10MMP = CIFAR10Tropical


class ImageNetTropical(nn.Module):
    """
    Tropical model for ImageNet.

    ResNet-34 style backbone + tropical classifier head.

    Args:
        num_classes: Number of classes.
        use_tropical: Use tropical classifier.
        dropout: Dropout rate.
    """

    def __init__(self, num_classes: int = 1000, use_tropical: bool = True, dropout: float = 0.2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stages
        self.stage1 = self._make_stage(64, 64, 3)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if use_tropical:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                MaxPlusAffine(1024),
                MinPlusAffine(1024),
                nn.Dropout(dropout),
                nn.Linear(1024, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(1024, num_classes),
            )

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int = 1):
        layers = []
        if stride != 1 or in_ch != out_ch:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ))
        for _ in range(num_blocks):
            layers.append(ResBlock(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Alias
ImageNetMMP = ImageNetTropical


class MMPConvClassifier(nn.Module):
    """Conv network with tropical classifier head."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        use_tropical: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Conv stages
        self.stage1 = ConvBlock(in_channels, base_channels)
        self.stage2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = base_channels * 8

        if use_tropical:
            self.classifier = nn.Sequential(
                TropicalBlock(feature_dim, 256, dropout=dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)


class MMPResNet(nn.Module):
    """ResNet with tropical classifier head."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        use_tropical: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_dim = base_channels * 8

        if use_tropical:
            self.classifier = nn.Sequential(
                TropicalBlock(feature_dim, 256, dropout=dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int, stride: int = 1):
        layers = []
        if stride != 1 or in_ch != out_ch:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ))
        for _ in range(num_blocks):
            layers.append(ResBlock(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)


def create_cifar10_model(model_type: str = "tropical", **kwargs):
    """Factory for CIFAR-10 models."""
    if model_type in ("tropical", "mmp"):
        return CIFAR10Tropical(use_tropical=True, **kwargs)
    elif model_type == "baseline":
        return CIFAR10Tropical(use_tropical=False, **kwargs)
    elif model_type == "resnet_tropical":
        return MMPResNet(num_classes=10, use_tropical=True, **kwargs)
    elif model_type == "resnet_baseline":
        return MMPResNet(num_classes=10, use_tropical=False, **kwargs)
    else:
        raise ValueError(f"Unknown: {model_type}")


def create_imagenet_model(model_type: str = "tropical", **kwargs):
    """Factory for ImageNet models."""
    if model_type in ("tropical", "mmp"):
        return ImageNetTropical(use_tropical=True, **kwargs)
    elif model_type == "baseline":
        return ImageNetTropical(use_tropical=False, **kwargs)
    else:
        raise ValueError(f"Unknown: {model_type}")


__all__ = [
    "ConvBlock",
    "ResBlock",
    "CIFAR10Tropical",
    "ImageNetTropical",
    "MMPConvClassifier",
    "MMPResNet",
    "create_cifar10_model",
    "create_imagenet_model",
    # Aliases
    "CIFAR10MMP",
    "ImageNetMMP",
]
