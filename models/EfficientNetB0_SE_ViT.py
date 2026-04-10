import math

import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


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


class EfficientNetB0_SE_ViT(nn.Module):
    """EfficientNet-B0 + Squeeze-Excite + 1-layer ViT encoder across slices."""

    def __init__(self, nhead: int = 8, dropout: float = 0.1, use_cls_token: bool = True):
        super().__init__()

        self.backbone = _build_efficientnet_b0().features
        self.se = SqueezeExcite(1280, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.use_cls_token = use_cls_token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(1280)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1280))
        else:
            self.register_parameter("cls_token", None)

        self.fc = nn.Sequential(
            nn.Linear(3 * 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def _positional_encoding(self, length: int, device, dtype):
        # Sinusoidal positional encoding: [1, length, 1280]
        d_model = 1280
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _encode_plane(self, x):
        # x: [S, C, H, W] or [B, S, C, H, W]
        squeeze_back = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = self.backbone(x)
        feat = self.se(feat)
        feat = self.pool(feat).view(b, s, -1)  # [B, S, 1280]

        pos = self._positional_encoding(s, feat.device, feat.dtype)
        tokens = feat + pos

        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.vit(tokens)
        tokens = self.norm(tokens)

        if self.use_cls_token:
            plane_feat = tokens[:, 0, :]
        else:
            plane_feat = tokens.mean(dim=1)

        if squeeze_back:
            plane_feat = plane_feat.view(1, -1)
        return plane_feat

    def forward(self, x):
        images = x
        axial = self._encode_plane(images[0])
        coronal = self._encode_plane(images[1])
        sagittal = self._encode_plane(images[2])

        feats = torch.cat([axial, coronal, sagittal], dim=1)
        output = self.fc(feats)
        return output
