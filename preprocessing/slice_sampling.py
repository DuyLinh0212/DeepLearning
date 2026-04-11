import numpy as np
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

    # Downsample by uniform index selection (no interpolation)
    if num_slices > target_slices:
        idx = np.linspace(0, num_slices - 1, target_slices)
        idx = np.round(idx).astype(int)
        return volume[idx]

    # Upsample by repeating the best slice to reach target_slices
    # "Best" slice is the one with highest variance (most texture/information).
    scores = np.var(volume.reshape(num_slices, -1), axis=1)
    best_idx = int(np.argmax(scores)) if num_slices > 0 else 0
    repeat_count = target_slices - num_slices
    if repeat_count <= 0:
        return volume
    repeated = np.repeat(volume[best_idx:best_idx + 1], repeat_count, axis=0)
    return np.concatenate([volume, repeated], axis=0)
