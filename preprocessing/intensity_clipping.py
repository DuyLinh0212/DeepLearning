import numpy as np


def percentile_clipping(volume: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5) -> np.ndarray:
    """Loai bo nhieu cuong do bang percentile clipping.

    Args:
        volume (np.ndarray): Anh MRI dang (S, H, W) hoac (H, W)
        lower_percentile (float): Percentile duoi
        upper_percentile (float): Percentile tren

    Returns:
        np.ndarray: Anh da duoc clip
    """
    if volume.size == 0:
        return volume

    # Tính ngưỡng theo percentile
    lower_val = np.percentile(volume, lower_percentile)
    upper_val = np.percentile(volume, upper_percentile)

    # Cắt bỏ những giá trị ngoài ngưỡng
    clipped = np.clip(volume, lower_val, upper_val)
    return clipped
