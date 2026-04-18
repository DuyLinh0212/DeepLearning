import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VIEWS = ("axial", "sagittal", "coronal")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def normalize_case_id(raw_id: str) -> str:
    base = os.path.splitext(os.path.basename(str(raw_id)))[0]
    return base.zfill(4)


def read_label_csv(path: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sample_id, target = line.split(",")
            labels[normalize_case_id(sample_id)] = int(target)
    return labels


def match_target_slices(volume: np.ndarray, target_slices: int) -> np.ndarray:
    if target_slices <= 0:
        return volume
    current = volume.shape[0]
    if current == target_slices:
        return volume
    if current > target_slices:
        return volume[:target_slices]
    pad_count = target_slices - current
    pad = np.zeros((pad_count, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
    return np.concatenate([volume, pad], axis=0)


def to_three_channel(volume: np.ndarray) -> np.ndarray:
    # Keep baseline loading simple: only convert grayscale [S,H,W] -> [S,3,H,W].
    return np.stack([volume, volume, volume], axis=1).astype(np.float32)


def normalize_imagenet(volume_3ch: np.ndarray) -> np.ndarray:
    vmin = float(volume_3ch.min())
    vmax = float(volume_3ch.max())
    if vmax > vmin:
        volume_3ch = (volume_3ch - vmin) / (vmax - vmin)
    else:
        volume_3ch = np.zeros_like(volume_3ch, dtype=np.float32)
    volume_3ch = (volume_3ch - IMAGENET_MEAN) / IMAGENET_STD
    return volume_3ch.astype(np.float32)


class MRNetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_dir: str,
        split: str,
        task: str,
        target_slices: int = 32,
    ) -> None:
        self.split = split
        self.task = task
        self.target_slices = target_slices
        self.data_dir = os.path.join(data_dir, split)

        label_path = os.path.join(labels_dir, f"{split}-{task}.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing label file: {label_path}")

        label_map = read_label_csv(label_path)
        valid_ids: List[str] = []
        for case_id in label_map:
            has_all_views = all(
                os.path.exists(os.path.join(self.data_dir, view, f"{case_id}.npy"))
                for view in VIEWS
            )
            if has_all_views:
                valid_ids.append(case_id)

        if not valid_ids:
            raise RuntimeError(
                f"No valid samples for split={split}, task={task}. Checked directory: {self.data_dir}"
            )

        self.case_ids = sorted(valid_ids)
        self.labels = [label_map[case_id] for case_id in self.case_ids]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int):
        case_id = self.case_ids[index]
        volumes = []
        for view in VIEWS:
            volume_path = os.path.join(self.data_dir, view, f"{case_id}.npy")
            volume = np.load(volume_path).astype(np.float32)
            volume = match_target_slices(volume=volume, target_slices=self.target_slices)
            volume = to_three_channel(volume)
            volume = normalize_imagenet(volume)
            volumes.append(torch.from_numpy(volume))

        label = torch.tensor([float(self.labels[index])], dtype=torch.float32)
        return volumes[0], volumes[1], volumes[2], label, case_id


def mrnet_collate_fn(batch):
    axial_batch = [item[0] for item in batch]
    sagittal_batch = [item[1] for item in batch]
    coronal_batch = [item[2] for item in batch]
    labels = torch.cat([item[3] for item in batch], dim=0).view(-1, 1)
    case_ids = [item[4] for item in batch]
    return axial_batch, sagittal_batch, coronal_batch, labels, case_ids


def compute_pos_weight(labels: Sequence[int]) -> torch.Tensor:
    positives = float(sum(labels))
    negatives = float(len(labels) - positives)
    if positives <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([max(negatives / positives, 1e-6)], dtype=torch.float32)


def create_loaders(
    task: str,
    data_dir: str,
    labels_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    target_slices: int,
) -> Tuple[DataLoader, DataLoader]:
    train_set = MRNetDataset(
        data_dir=data_dir,
        labels_dir=labels_dir,
        split="train",
        task=task,
        target_slices=target_slices,
    )
    valid_set = MRNetDataset(
        data_dir=data_dir,
        labels_dir=labels_dir,
        split="valid",
        task=task,
        target_slices=target_slices,
    )

    max_workers = os.cpu_count() or 1
    workers = max(0, min(num_workers, max_workers))
    persistent_workers = workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=mrnet_collate_fn,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=mrnet_collate_fn,
    )
    return train_loader, valid_loader
