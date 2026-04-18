import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN    = 58.09
STDDEV  = 49.73
# ImageNet mean/std theo chuẩn torchvision (ảnh đã scale về [0,1])
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _preprocess_volume(vol, use_imagenet_norm):
    """
    Tiền xử lý một volume MRI (shape: [S, H, W]) thành tensor [S, 3, 224, 224].

    Pipeline:
      1. Crop trung tâm về INPUT_DIM x INPUT_DIM
      2. Min-max normalize về [0, 1]
      3a. ImageNet path  (efficientnet): chuẩn hóa theo IMAGENET_MEAN/STD
      3b. AlexNet path   (alexnet/resnet): nhân 255 → chuẩn hóa theo MEAN/STDDEV
      4. Stack thành 3 kênh
    """
    # 1. Crop
    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    if pad > 0:
        vol = vol[:, pad:-pad, pad:-pad]

    # 2. Min-max → [0, 1]
    v_min, v_max = vol.min(), vol.max()
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = vol * 0.0  # ảnh đồng nhất

    if use_imagenet_norm:
        # 3a. Stack trước rồi chuẩn hóa theo ImageNet (giá trị đã trong [0,1])
        vol = np.stack((vol,) * 3, axis=1)          # [S, 3, H, W]
        mean = np.array(IMAGENET_MEAN)[None, :, None, None]
        std  = np.array(IMAGENET_STD) [None, :, None, None]
        vol  = (vol - mean) / std
    else:
        # 3b. Scale về [0,255] rồi chuẩn hóa theo thống kê MRNet gốc
        vol = vol * MAX_PIXEL_VAL
        vol = (vol - MEAN) / STDDEV
        vol = np.stack((vol,) * 3, axis=1)          # [S, 3, H, W]

    return vol


class Dataset(data.Dataset):
    def __init__(self, data_dir, labels_dir, split, tear_type, use_gpu,
                 backbone="alexnet"):
        super().__init__()
        self.use_gpu = use_gpu
        self.backbone = backbone
        self.use_imagenet_norm = backbone.startswith("efficientnet")

        def _norm_id(raw_id):
            base = os.path.splitext(os.path.basename(str(raw_id)))[0]
            return base.zfill(4)

        split = split.lower()
        self.datadir = os.path.join(data_dir, split)

        # --- Đọc nhãn chính ---
        label_dict = {}
        label_path = os.path.join(labels_dir, f"{split}-{tear_type}.csv")
        for line in open(label_path).readlines():
            parts = line.strip().split(',')
            label_dict[_norm_id(parts[0])] = int(parts[1])

        # --- Đọc nhãn abnormal (dùng để tính class weight) ---
        abnormal_label_dict = {}
        abnormal_path = os.path.join(labels_dir, f"{split}-abnormal.csv")
        if os.path.exists(abnormal_path):
            for line in open(abnormal_path).readlines():
                parts = line.strip().split(',')
                abnormal_label_dict[_norm_id(parts[0])] = int(parts[1])

        # --- Xây danh sách path theo thứ tự label_dict ---
        self.paths = [f"{rid}.npy" for rid in label_dict.keys()]
        rids = [p.split(".")[0] for p in self.paths]

        self.labels = [label_dict[r] for r in rids]
        if abnormal_label_dict:
            self.abnormal_labels = [abnormal_label_dict[r] for r in rids]
        else:
            self.abnormal_labels = self.labels

        # --- Class weight cho weighted BCE ---
        if tear_type != "abnormal":
            temp = [self.labels[i] for i in range(len(self.labels))
                    if self.abnormal_labels[i] == 1]
            neg_weight = float(np.mean(temp)) if temp else 0.5
        else:
            neg_weight = float(np.mean(self.labels))
        self.weights = [neg_weight, 1.0 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy    = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        return F.binary_cross_entropy_with_logits(
            prediction, target, weight=Variable(weights_tensor))

    def __getitem__(self, index):
        filename = self.paths[index]
        vol_axial = np.load(os.path.join(self.datadir, "axial",     filename))
        vol_sagit = np.load(os.path.join(self.datadir, "sagittal",  filename))
        vol_coron = np.load(os.path.join(self.datadir, "coronal",   filename))

        vol_axial = _preprocess_volume(vol_axial, self.use_imagenet_norm)
        vol_sagit = _preprocess_volume(vol_sagit, self.use_imagenet_norm)
        vol_coron = _preprocess_volume(vol_coron, self.use_imagenet_norm)

        label_tensor = torch.FloatTensor([self.labels[index]])

        return (torch.FloatTensor(vol_axial),
                torch.FloatTensor(vol_sagit),
                torch.FloatTensor(vol_coron),
                label_tensor,
                self.abnormal_labels[index])

    def __len__(self):
        return len(self.paths)


def load_data(task="acl", use_gpu=False, data_dir="data", labels_dir="labels",
              num_workers=4, backbone="alexnet"):
    pin_memory = bool(use_gpu)
    max_workers = os.cpu_count() or 1
    num_workers = max(0, min(num_workers, max_workers, 4))

    train_dataset = Dataset(data_dir, labels_dir, "train", task, use_gpu, backbone)
    valid_dataset = Dataset(data_dir, labels_dir, "valid", task, use_gpu, backbone)
    test_dataset  = Dataset(data_dir, labels_dir, "test",  task, use_gpu, backbone)

    def _make_loader(dataset, shuffle):
        return data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0 and shuffle),
        )

    return (_make_loader(train_dataset, shuffle=True),
            _make_loader(valid_dataset, shuffle=False),
            _make_loader(test_dataset,  shuffle=False))
