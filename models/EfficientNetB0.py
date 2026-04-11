import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 backbone for 3-plane MR slices with max pooling over slices/planes."""

    def __init__(self):
        super().__init__()

        self.backbone = _build_efficientnet_b0().features
        self.se = SqueezeExcite(1280, reduction=16)

        # Feature dimension for EfficientNet-B0 is 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_norm = nn.LayerNorm(1280)
        self.feat_dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def _encode_plane(self, net, x):
        # x can be [S, C, H, W] or [B, S, C, H, W]
        if x.dim() == 4:
            feat = net(x)
            feat = self.se(feat)
            feat = self.pool(feat).view(feat.size(0), -1)
            feat = self.feat_norm(feat)
            feat = self.feat_dropout(feat)
            feat, _ = torch.max(feat, dim=0, keepdim=True)
            return feat

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.se(feat)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = feat.view(b, s, -1)
        feat, _ = torch.max(feat, dim=1)
        return feat

    def forward(self, x):
        # Expect list of 3 tensors, each: [B, S, C, H, W] or [S, C, H, W]
        images = x
        axial = self._encode_plane(self.backbone, images[0])
        coronal = self._encode_plane(self.backbone, images[1])
        sagittal = self._encode_plane(self.backbone, images[2])

        planes = torch.stack([axial, coronal, sagittal], dim=1)  # [B, 3, 1280]
        feats, _ = torch.max(planes, dim=1)
        output = self.fc(feats)
        return output


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale
