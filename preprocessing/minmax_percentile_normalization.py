import numpy as np


def minmax_percentile_normalize(
    volume: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5, eps: float = 1e-8
) -> np.ndarray:
    """Chuan hoa Min-Max sau khi percentile clipping.

    Args:
        volume (np.ndarray): Anh MRI dang (S, H, W)
        lower_percentile (float): Percentile duoi
        upper_percentile (float): Percentile tren
        eps (float): Gia tri nho de tranh chia cho 0

    Returns:
        np.ndarray: Anh da duoc chuan hoa ve [0, 1]
    """
    if volume.size == 0:
        return volume

    lower_val = np.percentile(volume, lower_percentile)
    upper_val = np.percentile(volume, upper_percentile)
    if upper_val <= lower_val:
        upper_val = lower_val + eps

    volume = np.clip(volume, lower_val, upper_val)
    return (volume - lower_val) / (upper_val - lower_val + eps)
