from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def resize_volume_bilinear(volume: np.ndarray, target_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Resize MRI ve cung kich thuoc bang Bilinear Interpolation.

    Args:
        volume (np.ndarray): Anh MRI dang (S, H, W)
        target_size (int | tuple[int, int]): Kich thuoc dich (H, W)

    Returns:
        np.ndarray: Anh da duoc resize
    """
    if volume.size == 0:
        return volume

    if isinstance(target_size, int):
        size_hw = (target_size, target_size)
    else:
        size_hw = target_size

    # Chuyen sang tensor de dung bilinear interpolate
    tensor = torch.from_numpy(volume).unsqueeze(1).float()  # (S, 1, H, W)
    resized = F.interpolate(tensor, size=size_hw, mode='bilinear', align_corners=False)
    return resized.squeeze(1).numpy()
