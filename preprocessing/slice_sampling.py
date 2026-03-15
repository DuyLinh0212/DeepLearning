import numpy as np


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

    # Linspace giup lay deu cac slice; neu thieu se lap lai slice
    indices = np.linspace(0, num_slices - 1, target_slices)
    indices = np.round(indices).astype(int)
    return volume[indices]
