import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data

from preprocessing.intensity_clipping import percentile_clipping
from preprocessing.z_score_normalization import z_score_normalize
from preprocessing.resize import resize_volume_bilinear
from preprocessing.slice_sampling import uniform_slice_sampling
from preprocessing.augmentation import random_augmentation


class MRDataset(data.Dataset):
    def __init__(
        self,
        task: str = 'acl',
        split: str = 'train',
        data_root: str = './data',
        label_root: str = './labels',
        target_slices: int = 32,
        image_size: int = 224,
        augment: bool = False,
    ):
        super().__init__()
        self.planes = ['axial', 'coronal', 'sagittal']
        self.task = task
        self.split = split
        self.data_root = data_root
        self.label_root = label_root
        self.target_slices = target_slices
        self.image_size = image_size
        self.augment = augment

        # Doc danh sach label
        self.records = None
        self.labels = None
        self.has_label = True

        label_path = os.path.join(self.label_root, f'{split}-{task}.csv')
        if os.path.exists(label_path):
            self.records = pd.read_csv(label_path, header=None, names=['id', 'label'])
            self.labels = self.records['label'].tolist()
        else:
            # Neu khong co label (vi du: test), tao danh sach rong
            self.has_label = False
            self.records = None
            self.labels = []

        # Dinh dang id thanh 4 ky tu
        if self.records is not None:
            self.records['id'] = self.records['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))

        # Tao duong dan file .npy cho moi plane
        self.paths = {}
        for plane in self.planes:
            plane_dir = os.path.join(self.data_root, split, plane)
            if self.records is not None:
                self.paths[plane] = [os.path.join(plane_dir, f'{filename}.npy') for filename in self.records['id']]
            else:
                # Neu khong co label, lay toan bo file trong thu muc
                if os.path.isdir(plane_dir):
                    self.paths[plane] = sorted([
                        os.path.join(plane_dir, f) for f in os.listdir(plane_dir) if f.endswith('.npy')
                    ])
                else:
                    self.paths[plane] = []

        # Tinh weight cho loss (chi dung khi co label)
        if self.has_label and len(self.labels) > 0:
            pos = sum(self.labels)
            neg = len(self.labels) - pos
            self.pos_weight = torch.FloatTensor([neg / max(pos, 1)])
        else:
            self.pos_weight = torch.FloatTensor([1.0])

        print(f'Task: {task} | Split: {split} | Samples: {self.__len__()}')

    def __len__(self):
        if self.records is not None:
            return len(self.records)
        # Neu khong co label, lay do dai theo so file o plane axial (neu co)
        return len(self.paths.get('axial', []))

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            volume = np.load(self.paths[plane][index])
            img_raw[plane] = self._preprocess_volume(volume, augment=self.augment)

        if self.has_label:
            label = self.labels[index]
            label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])
        else:
            # Neu test khong co label, tra ve 0 de giu dung format
            label = torch.FloatTensor([0])

        return [img_raw[plane] for plane in self.planes], label

    def _preprocess_volume(self, volume: np.ndarray, augment: bool = False) -> torch.Tensor:
        # 1. Percentile Clipping (loai bo nhieu cuong do)
        volume = percentile_clipping(volume)

        # 2. Z-score normalization (chuan hoa cuong do pixel)
        volume = z_score_normalize(volume)

        # 3. Uniform slice sampling (dua ve so slice co dinh)
        volume = uniform_slice_sampling(volume, target_slices=self.target_slices)

        # 4. Resize bang bilinear interpolation (dua ve cung kich thuoc)
        volume = resize_volume_bilinear(volume, target_size=self.image_size)

        # 5. Chuyen sang Tensor
        volume_tensor = torch.from_numpy(volume).float()  # (S, H, W)

        # 6. Augmentation (chi cho train)
        if augment:
            volume_tensor = random_augmentation(volume_tensor)

        # 7. Tao 3 kenh mau (RGB gia) de phu hop backbone
        volume_tensor = volume_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

        return volume_tensor


def load_data(
    task: str,
    batch_size: int = 1,
    num_workers: int = 0,
    target_slices: int = 32,
    image_size: int = 224,
):
    # Tao Dataset cho train/valid/test
    train_data = MRDataset(
        task=task,
        split='train',
        target_slices=target_slices,
        image_size=image_size,
        augment=True,
    )
    valid_data = MRDataset(
        task=task,
        split='valid',
        target_slices=target_slices,
        image_size=image_size,
        augment=False,
    )
    test_data = MRDataset(
        task=task,
        split='test',
        target_slices=target_slices,
        image_size=image_size,
        augment=False,
    )

    # Tao DataLoader
    train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader, test_loader
