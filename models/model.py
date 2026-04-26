"""
Dual-Stream Forgery Detection Model
=====================================
Architecture:
  Stream A  →  EfficientNet-B3 backbone  →  processes original image
  Stream B  →  EfficientNet-B3 backbone  →  processes ELA image
  Fusion    →  Concat + MLP head         →  binary classification

Why dual-stream?
  The original image captures semantic context; ELA captures compression
  artifacts.  Fusing both gives the model complementary evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─── Attention Fusion Module ────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ─── Single Stream ──────────────────────────────────────────────────────────────

class StreamEncoder(nn.Module):
    """
    One EfficientNet-B3 backbone + channel attention.
    Outputs a feature vector of size `feature_dim`.
    """

    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        backbone = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=0
        )
        self.features = backbone          # outputs (B, 1536, 10, 10) for 224×224 input
        self.attn = ChannelAttention(backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.num_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        feat = self.features.forward_features(x)  # (B, C, H, W)
        feat = self.attn(feat)
        feat = self.pool(feat)
        return self.proj(feat)


# ─── Dual-Stream Model ──────────────────────────────────────────────────────────

class DualStreamForgeryDetector(nn.Module):
    """
    Main model combining original-image stream and ELA stream.

    Args:
        pretrained   : Use ImageNet pretrained weights.
        feature_dim  : Per-stream embedding size.
        num_classes  : 2 for binary (authentic / forged).
        dropout      : Dropout rate in the classifier head.
    """

    def __init__(self, pretrained: bool = True, feature_dim: int = 512,
                 num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.stream_orig = StreamEncoder(pretrained, feature_dim)
        self.stream_ela  = StreamEncoder(pretrained, feature_dim)

        fused_dim = feature_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, orig: torch.Tensor, ela: torch.Tensor):
        """
        Args:
            orig : (B, 3, H, W) original document image tensor.
            ela  : (B, 3, H, W) ELA image tensor.

        Returns:
            logits : (B, num_classes)
        """
        f_orig = self.stream_orig(orig)
        f_ela  = self.stream_ela(ela)
        fused  = torch.cat([f_orig, f_ela], dim=1)
        return self.classifier(fused)

    def predict_proba(self, orig: torch.Tensor, ela: torch.Tensor):
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(orig, ela)
            return F.softmax(logits, dim=1)


# ─── Lightweight Single-Stream Fallback ─────────────────────────────────────────

class SingleStreamDetector(nn.Module):
    """
    Simpler ResNet-50 based model for quick experiments or limited compute.
    Takes only the ELA image as input.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        backbone = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=0
        )
        self.features = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, ela: torch.Tensor):
        feat = self.features.forward_features(ela)
        feat = F.adaptive_avg_pool2d(feat, 1)
        return self.classifier(feat)


# ─── Model Factory ──────────────────────────────────────────────────────────────

def build_model(model_type: str = "dual", pretrained: bool = True,
                num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function.

    Args:
        model_type : 'dual' or 'single'.
        pretrained : ImageNet weights.
        num_classes: Number of output classes.

    Returns:
        model : nn.Module
    """
    if model_type == "dual":
        return DualStreamForgeryDetector(pretrained=pretrained,
                                         num_classes=num_classes, **kwargs)
    elif model_type == "single":
        return SingleStreamDetector(pretrained=pretrained,
                                    num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'dual' or 'single'.")
