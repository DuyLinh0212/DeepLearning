import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 backbone for 3-plane MR slices with attention pooling over slices."""

    def __init__(self):
        super().__init__()

        self.backbone = _build_efficientnet_b0().features

        # Feature dimension for EfficientNet-B0 is 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Linear(1280, 1)
        self.fc = nn.Sequential(
            nn.Linear(3 * 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def _encode_plane(self, net, x):
        # x can be [S, C, H, W] or [B, S, C, H, W]
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)
            attn = self.attn(feat)  # [S, 1]
            weights = torch.softmax(attn.squeeze(1), dim=0).view(-1, 1)
            feat = torch.sum(feat * weights, dim=0, keepdim=True)
            return feat

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(b, s, -1)
        attn = self.attn(feat)  # [B, S, 1]
        weights = torch.softmax(attn.squeeze(2), dim=1).unsqueeze(2)
        feat = torch.sum(feat * weights, dim=1)
        return feat

    def forward(self, x):
        # Expect list of 3 tensors, each: [B, S, C, H, W] or [S, C, H, W]
        images = x
        axial = self._encode_plane(self.backbone, images[0])
        coronal = self._encode_plane(self.backbone, images[1])
        sagittal = self._encode_plane(self.backbone, images[2])

        feats = torch.cat([axial, coronal, sagittal], dim=1)
        output = self.fc(feats)
        return output
