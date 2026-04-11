import random
from typing import Optional

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def _apply_per_slice(volume: torch.Tensor, fn):
    # Ap dung cung 1 phep bien doi cho tat ca slice
    slices = []
    for i in range(volume.shape[0]):
        slice_i = volume[i].unsqueeze(0)  # (1, H, W)
        slice_i = fn(slice_i)
        slices.append(slice_i.squeeze(0))
    return torch.stack(slices, dim=0)


def _random_spatial_augmentation(volume: torch.Tensor) -> torch.Tensor:
    ops = ["rotate", "hflip", "crop"]
    op = random.choice(ops)

    if op == "rotate":
        angle = random.uniform(-12, 12)
        return _apply_per_slice(volume, lambda x: TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR))

    if op == "hflip":
        return _apply_per_slice(volume, TF.hflip)

    if op == "crop":
        _, h, w = volume.shape
        crop_h = int(h * 0.92)
        crop_w = int(w * 0.92)
        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0

        def _crop_and_resize(x):
            x = TF.crop(x, top, left, crop_h, crop_w)
            x = TF.resize(x, [h, w], interpolation=InterpolationMode.BILINEAR)
            return x

        return _apply_per_slice(volume, _crop_and_resize)

    return volume


def _random_intensity_augmentation(volume: torch.Tensor) -> torch.Tensor:
    # Assume volume is normalized to [0, 1]
    ops = ["brightness", "contrast", "gamma", "noise"]
    op = random.choice(ops)

    if op == "brightness":
        shift = random.uniform(-0.08, 0.08)
        volume = volume + shift
        return volume.clamp(0.0, 1.0)

    if op == "contrast":
        factor = random.uniform(0.9, 1.1)
        mean = volume.mean()
        volume = (volume - mean) * factor + mean
        return volume.clamp(0.0, 1.0)

    if op == "gamma":
        gamma = random.uniform(0.9, 1.1)
        volume = torch.clamp(volume, 0.0, 1.0)
        return torch.pow(volume, gamma)

    if op == "noise":
        sigma = random.uniform(0.0, 0.03)
        volume = volume + torch.randn_like(volume) * sigma
        return volume.clamp(0.0, 1.0)

    return volume


def random_augmentation(volume: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """Tang cuong du lieu bang phep bien doi ngau nhien.

    Args:
        volume (torch.Tensor): Tensor dang (S, H, W) da duoc normalize [0, 1]
        seed (int | None): Dat seed neu can lap lai

    Returns:
        torch.Tensor: Tensor sau augmentation
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if volume.dtype != torch.float32:
        volume = volume.float()

    # Spatial augmentation with 70% prob
    if random.random() < 0.7:
        volume = _random_spatial_augmentation(volume)

    # Intensity augmentation with 60% prob
    if random.random() < 0.6:
        volume = _random_intensity_augmentation(volume)

    return volume
