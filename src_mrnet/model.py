from typing import List, Sequence, Union

import torch
import torch.nn as nn
from torchvision import models


class SliceEncoderEfficientNetB0(nn.Module):
    def __init__(self, pretrained: bool = True, projected_dim: int = 256) -> None:
        super().__init__()
        if hasattr(models, "EfficientNet_B0_Weights"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        else:
            backbone = models.efficientnet_b0(pretrained=pretrained)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_projection = nn.Linear(1280, projected_dim)
        self.out_dim = projected_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.feature_projection(x)
        return x


class TripleMRNetEfficientNetB0(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.4, projected_dim: int = 256) -> None:
        super().__init__()
        self.axial_encoder = SliceEncoderEfficientNetB0(pretrained=pretrained, projected_dim=projected_dim)
        self.sagittal_encoder = SliceEncoderEfficientNetB0(pretrained=pretrained, projected_dim=projected_dim)
        self.coronal_encoder = SliceEncoderEfficientNetB0(pretrained=pretrained, projected_dim=projected_dim)
        self.post_pool_dropout = nn.Dropout(dropout)

        feature_dim = self.axial_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    @staticmethod
    def _to_volume_list(volumes: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        if isinstance(volumes, torch.Tensor):
            if volumes.ndim == 4:
                return [volumes]
            if volumes.ndim == 5:
                return [volumes[idx] for idx in range(volumes.shape[0])]
            raise ValueError(f"Unexpected tensor shape for volume input: {tuple(volumes.shape)}")
        return list(volumes)

    @staticmethod
    def _pool_over_slices(slice_features: torch.Tensor) -> torch.Tensor:
        max_features = torch.max(slice_features, dim=0, keepdim=True).values
        avg_features = torch.mean(slice_features, dim=0, keepdim=True)
        return 0.5 * (max_features + avg_features)

    def _encode_volume(self, volume: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        if volume.ndim == 5 and volume.shape[0] == 1:
            volume = volume.squeeze(0)
        if volume.ndim != 4:
            raise ValueError(f"Volume must be [S, 3, H, W], received shape {tuple(volume.shape)}")
        slice_features = encoder(volume)
        pooled_features = self._pool_over_slices(slice_features)
        return self.post_pool_dropout(pooled_features)

    def forward(
        self,
        vol_axial: Union[torch.Tensor, Sequence[torch.Tensor]],
        vol_sagittal: Union[torch.Tensor, Sequence[torch.Tensor]],
        vol_coronal: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        axial_list = self._to_volume_list(vol_axial)
        sagittal_list = self._to_volume_list(vol_sagittal)
        coronal_list = self._to_volume_list(vol_coronal)

        if not (len(axial_list) == len(sagittal_list) == len(coronal_list)):
            raise ValueError("Batch size mismatch across axial/sagittal/coronal inputs.")

        logits = []
        for axial_vol, sagittal_vol, coronal_vol in zip(axial_list, sagittal_list, coronal_list):
            f_axial = self._encode_volume(axial_vol, self.axial_encoder)
            f_sagittal = self._encode_volume(sagittal_vol, self.sagittal_encoder)
            f_coronal = self._encode_volume(coronal_vol, self.coronal_encoder)
            fused = torch.cat([f_axial, f_sagittal, f_coronal], dim=1)
            logits.append(self.classifier(fused))
        return torch.cat(logits, dim=0)
