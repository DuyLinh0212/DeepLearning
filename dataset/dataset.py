import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

# Match MRNet src loader behavior (center crop + MRNet normalization)

INPUT_DIM = 224

class MRData(data.Dataset):
    def __init__(self, task='acl', train=True, transform=None, weights=None, target_slices=32, input_dim=INPUT_DIM):
        super().__init__()
        self.planes = ['axial', 'coronal', 'sagittal']
        self.records = None
        self.image_path = {}
        self.target_slices = target_slices
        self.input_dim = input_dim
        self.train = train

        if train:
            self.records = pd.read_csv('./labels/train-{}.csv'.format(task), header=None, names=['id', 'label'])
            for plane in self.planes:
                self.image_path[plane] = './data/train/{}/'.format(plane)
        else:
            self.records = pd.read_csv('./labels/valid-{}.csv'.format(task), header=None, names=['id', 'label'])
            for plane in self.planes:
                self.image_path[plane] = './data/valid/{}/'.format(plane)

        self.transform = transform
        self.records['id'] = self.records['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))
        
        self.paths = {}
        for plane in self.planes:
            self.paths[plane] = [self.image_path[plane] + filename + '.npy' for filename in self.records['id'].tolist()]

        self.labels = self.records['label'].tolist()
        
        # T??nh to??n weight
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        if weights:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor([neg / pos])
        
        print(f'Task: {task} | Train: {train}')
        print(f'Samples: -ve: {neg}, +ve: {pos} | Loss Weights: {self.weights}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            # src loader does not resample slices
            img_raw[plane] = self._resize_image(img_raw[plane])
            
        label = self.labels[index]
        label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])

        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        # 1. Center crop to target size (match src loader)
        target = self.input_dim
        if target is not None and target <= image.shape[1] and target <= image.shape[2]:
            pad = int((image.shape[2] - target) / 2)
            image = image[:, pad:-pad, pad:-pad]

        # 2. Min-max normalize to [0, 255] (match src loader)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        denom = max(max_val - min_val, 1e-6)
        image = (image - min_val) / denom * 255.0

        # 3. To tensor & 3-channel
        image = torch.FloatTensor(image)
        image = torch.stack((image,) * 3, axis=1)

        # 5. Apply transform (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        return image

def load_data(task: str, batch_size: int = 1, num_workers: int = 0, target_slices: int = 32, image_size: int = INPUT_DIM):
    # ?????nh ngh??a Augmentation
    # L??u ??: Kh??ng c???n b?????c repeat/permute n???a v?? ???? l??m trong _resize_image
    mrnet_mean = 58.09
    mrnet_std = 49.73
    augments = transforms.Compose(
        [
            transforms.Normalize(
                mean=[mrnet_mean, mrnet_mean, mrnet_mean],
                std=[mrnet_std, mrnet_std, mrnet_std],
            ),
        ]
    )

    print('Loading Train Dataset of {} task...'.format(task))
    train_data = MRData(task, train=True, transform=augments, target_slices=target_slices, input_dim=image_size)
    # Weighted sampling to balance classes in each batch
    labels = train_data.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # num_workers=0 ????? tr??nh l???i tr??n Windows
    train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    print('Loading Validation Dataset of {} task...'.format(task))
    val_data = MRData(task, train=False, transform=augments, target_slices=target_slices, input_dim=image_size)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, train_data.weights, val_data.weights
