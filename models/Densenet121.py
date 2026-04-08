import torch
import torch.nn as nn
from torchvision import models


def _build_densenet121():
    try:
        return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    except Exception:
        return models.densenet121(pretrained=True)


class Densenet121(nn.Module):
    """DenseNet121 backbone for 3-plane MR slices (no extra optimization)."""

    def __init__(self):
        super().__init__()

        self.axial = _build_densenet121().features
        self.coronal = _build_densenet121().features
        self.sagittal = _build_densenet121().features

        # Feature dimension for DenseNet121 is 1024
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3 * 1024, 3)

    def _encode_plane(self, net, x):
        # x can be [S, C, H, W] or [B, S, C, H, W]
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)
            feat = torch.max(feat, dim=0, keepdim=True)[0]
            return feat

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(b, s, -1)
        feat = torch.max(feat, dim=1)[0]
        return feat

    def forward(self, x):
        # Expect list of 3 tensors, each: [B, S, C, H, W] or [S, C, H, W]
        images = x
        axial = self._encode_plane(self.axial, images[0])
        coronal = self._encode_plane(self.coronal, images[1])
        sagittal = self._encode_plane(self.sagittal, images[2])

        feats = torch.cat([axial, coronal, sagittal], dim=1)
        output = self.fc(feats)
        return output
