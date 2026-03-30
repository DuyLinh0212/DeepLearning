import numpy as np
import torch
import torch.nn.functional as F


def uniform_slice_sampling(volume: np.ndarray, target_slices: int = 32) -> np.ndarray:
    """Chon deu cac slice ve so luong co dinh.

    Args:
        volume (np.ndarray): Anh MRI dang (S, H, W)
        target_slices (int): So slice can lay

    Returns:
        np.ndarray: Anh da duoc chon slice
    """
    if volume.size == 0:
        return volume

    num_slices = volume.shape[0]
    if num_slices == target_slices:
        return volume

    # Interpolate along slice axis to reach target_slices
    # volume: (S, H, W) -> (1, 1, S, H, W)
    vol = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    vol = F.interpolate(
        vol,
        size=(target_slices, volume.shape[1], volume.shape[2]),
        mode="trilinear",
        align_corners=False,
    )
    vol = vol.squeeze(0).squeeze(0).cpu().numpy()
    return vol
