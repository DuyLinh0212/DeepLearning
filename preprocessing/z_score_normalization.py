import numpy as np


def z_score_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Chuan hoa z-score de on dinh qua trinh hoc.

    Args:
        volume (np.ndarray): Anh MRI dang (S, H, W) hoac (H, W)
        eps (float): Gia tri nho tranh chia cho 0

    Returns:
        np.ndarray: Anh da chuan hoa
    """
    if volume.size == 0:
        return volume

    # Dung float32 de giam bo nho
    volume = volume.astype(np.float32, copy=False)
    mean = np.mean(volume, dtype=np.float32)
    std = np.std(volume, dtype=np.float32)

    if std < eps:
        # Neu anh dong nhat thi tra ve 0
        return np.zeros_like(volume)

    return (volume - mean) / (std + eps)
