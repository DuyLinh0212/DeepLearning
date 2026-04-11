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

    # Downsample by uniform index selection (no interpolation)
    if num_slices > target_slices:
        idx = np.linspace(0, num_slices - 1, target_slices)
        idx = np.round(idx).astype(int)
        return volume[idx]

    # Upsample by repeating the best slice to reach target_slices
    # Hybrid score: center proximity + variance.
    variances = np.var(volume.reshape(num_slices, -1), axis=1)
    if np.max(variances) > 0:
        variances = variances / np.max(variances)
    center = (num_slices - 1) / 2.0
    center_score = 1.0 - (np.abs(np.arange(num_slices) - center) / (center + 1e-8))
    scores = 0.5 * variances + 0.5 * center_score
    best_idx = int(np.argmax(scores)) if num_slices > 0 else 0
    repeat_count = target_slices - num_slices
    if repeat_count <= 0:
        return volume
    repeated = np.repeat(volume[best_idx:best_idx + 1], repeat_count, axis=0)
    return np.concatenate([volume, repeated], axis=0)


def resample_slices_trilinear(volume: np.ndarray, target_slices: int = 32) -> np.ndarray:
    """Resample along slice axis using trilinear interpolation.

    Args:
        volume (np.ndarray): MRI volume shape (S, H, W)
        target_slices (int): Desired number of slices

    Returns:
        np.ndarray: Resampled volume with shape (target_slices, H, W)
    """
    if volume.size == 0:
        return volume

    num_slices, h, w = volume.shape
    if num_slices == target_slices:
        return volume

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()  # (1, 1, S, H, W)
    resized = F.interpolate(tensor, size=(target_slices, h, w), mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def probabilistic_slice_sampling(
    volume: np.ndarray,
    target_slices: int = 32,
    training: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Content-aware slice sampling using variance + center proximity weights.

    - If num_slices > target_slices: select slices by weighted sampling.
    - If num_slices < target_slices: resample by interpolation.

    Args:
        volume (np.ndarray): MRI volume shape (S, H, W)
        target_slices (int): Desired number of slices
        training (bool): Use stochastic sampling when True
        seed (int | None): Optional random seed for reproducibility

    Returns:
        np.ndarray: Volume with shape (target_slices, H, W)
    """
    if volume.size == 0:
        return volume

    if seed is not None:
        np.random.seed(seed)

    num_slices = volume.shape[0]
    if num_slices == target_slices:
        return volume

    if num_slices < target_slices:
        return resample_slices_trilinear(volume, target_slices)

    # Downsample with content-aware weights
    variances = np.var(volume.reshape(num_slices, -1), axis=1)
    if np.max(variances) > 0:
        variances = variances / np.max(variances)
    center = (num_slices - 1) / 2.0
    center_score = 1.0 - (np.abs(np.arange(num_slices) - center) / (center + 1e-8))

    weights = 0.6 * variances + 0.4 * center_score
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.sum(weights)

    if training:
        idx = np.random.choice(np.arange(num_slices), size=target_slices, replace=False, p=weights)
    else:
        idx = np.argsort(weights)[-target_slices:]

    idx = np.sort(idx)
    return volume[idx]
