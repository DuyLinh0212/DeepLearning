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


def random_augmentation(volume: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """Tang cuong du lieu bang 1 trong cac phep bien doi.

    Args:
        volume (torch.Tensor): Tensor dang (S, H, W)
        seed (int | None): Dat seed neu can lap lai

    Returns:
        torch.Tensor: Tensor sau augmentation
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    ops = ['rotate', 'hflip', 'crop', 'noise']
    op = random.choice(ops)

    if op == 'rotate':
        angle = random.uniform(-15, 15)
        return _apply_per_slice(volume, lambda x: TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR))

    if op == 'hflip':
        return _apply_per_slice(volume, TF.hflip)

    if op == 'crop':
        _, h, w = volume.shape
        crop_h = int(h * 0.9)
        crop_w = int(w * 0.9)
        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0

        def _crop_and_resize(x):
            x = TF.crop(x, top, left, crop_h, crop_w)
            x = TF.resize(x, [h, w], interpolation=InterpolationMode.BILINEAR)
            return x

        return _apply_per_slice(volume, _crop_and_resize)

    # Gaussian noise
    noise = torch.randn_like(volume) * 0.05
    return volume + noise
