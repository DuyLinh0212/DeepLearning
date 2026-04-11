import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from dataset.dataset import INPUT_DIM
# Match MRNet src loader behavior (center crop + MRNet normalization)
from config import config as base_config
from models import Densenet121, EfficientNetB0, EfficientNetB0_SE_ViT
from utils import _get_lr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _build_model(name: str):
    name = name.lower()
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
    if name == "efficientnetb0_se_vit":
        return EfficientNetB0_SE_ViT()
    raise ValueError(f"Unsupported model: {name}")


class MRDataByIds(Dataset):
    def __init__(self, ids, labels_map, data_dir="data", train=True, transform=None, target_slices=32, input_dim=INPUT_DIM):
        super().__init__()
        self.planes = ["axial", "coronal", "sagittal"]
        self.ids = list(ids)
        self.labels_map = labels_map
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.input_dim = input_dim
        self.train = train
        self.transform = transform

        self.paths = {p: [] for p in self.planes}
        for rid in self.ids:
            rid_str = str(rid).zfill(4)
            src_split = _find_source_split(self.data_dir, rid_str)
            if src_split is None:
                raise FileNotFoundError(f"Missing data for id={rid}")
            for plane in self.planes:
                self.paths[plane].append(os.path.join(self.data_dir, src_split, plane, f"{rid_str}.npy"))

        self.labels = [int(self.labels_map[rid]) for rid in self.ids]
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([neg / pos]) if pos > 0 else torch.FloatTensor([1.0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            # src loader does not resample slices
            img_raw[plane] = self._prepare_volume(img_raw[plane])

        label = self.labels[index]
        label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])
        return [img_raw[plane] for plane in self.planes], label

    def _prepare_volume(self, image):
        target = self.input_dim
        if target is not None and target <= image.shape[1] and target <= image.shape[2]:
            pad = int((image.shape[2] - target) / 2)
            image = image[:, pad:-pad, pad:-pad]

        min_val = float(np.min(image))
        max_val = float(np.max(image))
        denom = max(max_val - min_val, 1e-6)
        image = (image - min_val) / denom * 255.0

        # no augmentation (match src behavior)

        image = torch.FloatTensor(image)
        image = torch.stack((image,) * 3, axis=1)
        if self.transform:
            image = self.transform(image)
        return image


def _find_source_split(data_dir, rid_str):
    for split in ["train", "valid", "test"]:
        ok = True
        for plane in ["axial", "coronal", "sagittal"]:
            path = os.path.join(data_dir, split, plane, f"{rid_str}.npy")
            if not os.path.exists(path):
                ok = False
                break
        if ok:
            return split
    return None


def _read_split_labels(labels_dir, task, split):
    path = os.path.join(labels_dir, f"{split}-{task}.csv")
    data = np.genfromtxt(path, delimiter=",", dtype=int)
    if data.ndim == 1 and data.size == 2:
        data = np.array([data])
    ids = []
    labels_map = {}
    for rid, lab in data:
        rid = int(rid)
        labels_map[rid] = int(lab)
        ids.append(rid)
    return ids, labels_map


def _read_task_labels(labels_dir, task):
    labels_map = {}
    all_ids = []
    for split in ["train", "valid", "test"]:
        ids, split_map = _read_split_labels(labels_dir, task, split)
        all_ids.extend(ids)
        labels_map.update(split_map)
    return sorted(all_ids), labels_map


def _load_data_from_ids(ids_train, ids_val, labels_map, batch_size, num_workers, target_slices, image_size, data_dir):
    mrnet_mean = 58.09
    mrnet_std = 49.73
    norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[mrnet_mean, mrnet_mean, mrnet_mean],
                std=[mrnet_std, mrnet_std, mrnet_std],
            ),
        ]
    )

    train_data = MRDataByIds(
        ids_train,
        labels_map,
        data_dir=data_dir,
        train=True,
        transform=norm,
        target_slices=target_slices,
        input_dim=image_size,
    )
    labels = train_data.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    val_data = MRDataByIds(
        ids_val,
        labels_map,
        data_dir=data_dir,
        train=False,
        transform=norm,
        target_slices=target_slices,
        input_dim=image_size,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, train_data.weights, val_data.weights


def _load_test_from_ids(ids_test, labels_map, batch_size, num_workers, target_slices, image_size, data_dir):
    mrnet_mean = 58.09
    mrnet_std = 49.73
    norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[mrnet_mean, mrnet_mean, mrnet_mean],
                std=[mrnet_std, mrnet_std, mrnet_std],
            ),
        ]
    )
    test_data = MRDataByIds(
        ids_test,
        labels_map,
        data_dir=data_dir,
        train=False,
        transform=norm,
        target_slices=target_slices,
        input_dim=image_size,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader


def _best_threshold(y_true, y_prob):
    try:
        fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
        if len(thr) == 0:
            return 0.5
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])
    except Exception:
        return 0.5


def _weighted_bce_prob(prob, target, pos_weight=None, eps=1e-7):
    prob = prob.clamp(eps, 1.0 - eps)
    if pos_weight is None:
        return F.binary_cross_entropy(prob, target)
    return (-(pos_weight * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))).mean()


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    phase="train",
    threshold=0.5,
    auto_threshold=False,
    scaler=None,
    scheduler=None,
    use_amp=False,
    grad_clip_norm=None,
    abnormal_model=None,
    apply_abnormal_prior=False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    y_true = []
    y_prob = []
    losses = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, leave=False)

    for batch in iterator:
        if batch is None:
            continue
        images, label = batch

        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if use_amp and device != "cpu":
                with autocast():
                    output = model(images)
                    if abnormal_model is not None and apply_abnormal_prior:
                        with torch.no_grad():
                            abnormal_logits = abnormal_model(images)
                            abnormal_probs = torch.sigmoid(abnormal_logits)
                        probs = torch.sigmoid(output) * abnormal_probs
                        loss = _weighted_bce_prob(probs, label, getattr(criterion, "pos_weight", None))
                    else:
                        loss = criterion(output, label)
                if is_train:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(images)
                if abnormal_model is not None and apply_abnormal_prior:
                    with torch.no_grad():
                        abnormal_logits = abnormal_model(images)
                        abnormal_probs = torch.sigmoid(abnormal_logits)
                    probs = torch.sigmoid(output) * abnormal_probs
                    loss = _weighted_bce_prob(probs, label, getattr(criterion, "pos_weight", None))
                else:
                    loss = criterion(output, label)
                if is_train:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
            if is_train and scheduler is not None:
                scheduler.step()

        losses.append(loss.item())

        probas = torch.sigmoid(output)
        if abnormal_model is not None and (apply_abnormal_prior or not is_train):
            with torch.no_grad():
                abnormal_logits = abnormal_model(images)
                abnormal_probs = torch.sigmoid(abnormal_logits)
            probas = probas * abnormal_probs
        probas = probas.detach().cpu().numpy()
        labels = label.detach().cpu().numpy()

        y_prob.extend(probas.tolist())
        y_true.extend(labels.tolist())

    if len(losses) == 0:
        return 0.0, 0.5, 0.0, [], [], [], float(threshold)

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    if auto_threshold:
        threshold = _best_threshold(y_true, y_prob)

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    acc = metrics.accuracy_score(y_true, y_pred)
    loss_mean = float(np.mean(losses))
    return loss_mean, float(auc), float(acc), y_true, y_prob, y_pred, float(threshold)


def _append_csv(csv_path, row, header):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _plot_curves(csv_path, out_path):
    epochs = []
    train_loss = []
    val_loss = []
    train_auc = []
    val_auc = []
    train_acc = []
    val_acc = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(float(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_auc.append(float(row["train_auc"]))
                val_auc.append(float(row["val_auc"]))
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))
            except Exception:
                continue
    if len(epochs) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, train_auc, label="train_auc")
    plt.plot(epochs, val_auc, label="val_auc")
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, out_path):
    if len(y_true) == 0:
        return
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_roc(y_true, y_prob, out_path):
    if len(y_true) == 0:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
    except Exception:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _compute_confusion_metrics(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def _plot_confusion_matrix_multi(y_true, y_pred, class_names, out_dir, prefix):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    for i, name in enumerate(class_names):
        out_path = os.path.join(out_dir, f"{prefix}_{name}_confusion.png")
        _plot_confusion_matrix(y_true[:, i], y_pred[:, i], out_path)


def _plot_roc_multi(y_true, y_prob, class_names, out_path):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_prob = y_prob.reshape(-1, 1)
    plt.figure(figsize=(6, 6))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = metrics.roc_curve(y_true[:, i], y_prob[:, i])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} AUC = {auc:.4f}")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _compute_confusion_metrics_multi(y_true, y_pred, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    summary = {}
    accs = []
    precs = []
    sens = []
    specs = []
    f1s = []
    for i, name in enumerate(class_names):
        m = _compute_confusion_metrics(y_true[:, i], y_pred[:, i])
        summary[name] = m
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        sens.append(m["sensitivity"])
        specs.append(m["specificity"])
        f1s.append(m["f1"])
    summary["macro"] = {
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "sensitivity": float(np.mean(sens)) if sens else 0.0,
        "specificity": float(np.mean(specs)) if specs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
    return summary


def _set_trainable(model, train_backbone: bool):
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze head layers
    for name, module in model.named_modules():
        if name.endswith("fc") or name.endswith("attn"):
            for p in module.parameters():
                p.requires_grad = True

    # Unfreeze backbone if requested
    if train_backbone:
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = True
        else:
            for attr in ["axial", "coronal", "sagittal"]:
                if hasattr(model, attr):
                    for p in getattr(model, attr).parameters():
                        p.requires_grad = True


def _build_optimizer(model, lr, weight_decay, backbone_lr_mult=0.1):
    head_params = []
    backbone_params = []
    head_keys = ["fc", "attn", "plane_attn", "feat_norm", "feat_dropout", "se"]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keys):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def _build_onecycle(optimizer, max_lr, epochs, steps_per_epoch):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )


def train(
    config: dict,
    model_name: str,
    loaders=None,
    fold_idx=None,
    resume=True,
    threshold=0.5,
    labels_dir="labels",
    data_dir="data",
    run_test=True,
    device_arg="auto",
    abnormal_model_path=None,
):
    task_name = config.get("task", "acl")
    fold_suffix = f"fold_{fold_idx}" if fold_idx is not None else None
    save_folder = os.path.join("weights", task_name)
    if fold_suffix:
        save_folder = os.path.join(save_folder, fold_suffix)
    os.makedirs(save_folder, exist_ok=True)

    eval_folder = os.path.join("evaluation", f"{model_name}_{task_name}")
    if fold_suffix:
        eval_folder = os.path.join(eval_folder, fold_suffix)
    os.makedirs(eval_folder, exist_ok=True)

    csv_path = os.path.join(eval_folder, f"{model_name}_{task_name}_metrics.csv")
    best_model_path = os.path.join(save_folder, f"{model_name}_best_model.pth")
    last_model_path = os.path.join(save_folder, f"{model_name}_last_checkpoint.pth")

    print("Starting to Train Model...")
    if loaders is None:
        train_ids, train_map = _read_split_labels(labels_dir, task_name, "train")
        val_ids, val_map = _read_split_labels(labels_dir, task_name, "valid")
        labels_map = {**train_map, **val_map}
        train_loader, val_loader, train_wts, _ = _load_data_from_ids(
            train_ids,
            val_ids,
            labels_map,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            target_slices=config["target_slices"],
            image_size=config["image_size"],
            data_dir=data_dir,
        )
    else:
        train_loader, val_loader, train_wts, _ = loaders

    print("Initializing Model...")
    model = _build_model(model_name)
    if device_arg == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cpu":
        model = model.to(device)
        train_wts = train_wts.to(device)

    use_dataparallel = False
    if device == "cuda" and torch.cuda.device_count() > 1 and device_arg in ["auto", "cuda"]:
        model = torch.nn.DataParallel(model)
        use_dataparallel = True

    print("Initializing Loss Method...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    if device != "cpu":
        criterion = criterion.to(device)
        val_criterion = val_criterion.to(device)

    scaler = GradScaler(enabled=(device == "cuda"))
    abnormal_model = None
    if abnormal_model_path is not None:
        print(f"Loading abnormality prior model from {abnormal_model_path}")
        abnormal_model = _build_model(model_name)
        state = torch.load(abnormal_model_path, map_location=device)
        abnormal_model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
        if device != "cpu":
            abnormal_model = abnormal_model.to(device)
        abnormal_model.eval()

    apply_abnormal_prior = bool(config.get("abnormal_prior_in_train", True))

    starting_epoch = config["starting_epoch"]
    num_epochs = config["max_epoch"]
    best_val_auc = float(0)

    if resume and os.path.exists(last_model_path):
        print(f"Found checkpoint at {last_model_path}. Loading...")
        checkpoint = torch.load(last_model_path, map_location=device)
        if use_dataparallel:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        starting_epoch = checkpoint.get("epoch", starting_epoch) + 1
        best_val_auc = checkpoint.get("best_val_auc", best_val_auc)
        print(f"Resuming from epoch {starting_epoch} | Best AUC {best_val_auc:.4f}")

    if SummaryWriter is None:
        writer = _NullWriter()
    else:
        writer = SummaryWriter(comment=f"model={model_name} lr={config['lr']} task={task_name} fold={fold_idx}")
    t_start_training = time.time()

    header = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_acc",
        "val_loss",
        "val_auc",
        "val_acc",
        "best_threshold",
        "lr",
    ]

    _set_trainable(model, True)
    backbone_lr_mult = float(config.get("backbone_lr_mult", 0.1))
    optimizer = _build_optimizer(model, config["lr"], config["weight_decay"], backbone_lr_mult=backbone_lr_mult)
    if len(optimizer.param_groups) > 1:
        max_lr = [config["lr"], config["lr"] * backbone_lr_mult]
    else:
        max_lr = config["lr"]
    scheduler = _build_onecycle(optimizer, max_lr, num_epochs, len(train_loader))
    patience = config.get("patience", 5)
    epochs_no_improve = 0
    grad_clip_norm = config.get("clip_grad_norm", None)

    for epoch in range(starting_epoch, num_epochs):
        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        train_loss, train_auc, train_acc, _, _, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            threshold=threshold,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=(device == "cuda"),
            grad_clip_norm=grad_clip_norm,
            abnormal_model=abnormal_model,
            apply_abnormal_prior=apply_abnormal_prior,
        )
        val_loss, val_auc, val_acc, _, _, _, best_thresh = _run_epoch(
            model,
            val_loader,
            val_criterion,
            optimizer=None,
            device=device,
            phase="val",
            threshold=threshold,
            auto_threshold=True,
            scaler=scaler,
            use_amp=(device == "cuda"),
            abnormal_model=abnormal_model,
            apply_abnormal_prior=apply_abnormal_prior,
        )

        writer.add_scalar("Train/Avg Loss", train_loss, epoch)
        writer.add_scalar("Train/AUC_epoch", train_auc, epoch)
        writer.add_scalar("Train/Acc_epoch", train_acc, epoch)
        writer.add_scalar("Val/Avg Loss", val_loss, epoch)
        writer.add_scalar("Val/AUC_epoch", val_auc, epoch)
        writer.add_scalar("Val/Acc_epoch", val_acc, epoch)

        t_end = time.time()
        delta = t_end - epoch_start_time
        thresh_str = f"{best_thresh:.4f}"
        print(
            "Epoch [{}/{}] | train loss {:.4f} | train auc {:.4f} | train acc {:.4f} | "
            "val loss {:.4f} | val auc {:.4f} | val acc {:.4f} | thr [{}] | time {:.2f} s".format(
                epoch,
                num_epochs,
                train_loss,
                train_auc,
                train_acc,
                val_loss,
                val_auc,
                val_acc,
                thresh_str,
                delta,
            )
        )
        print("-" * 30)
        writer.flush()

        _append_csv(
            csv_path,
            [epoch, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, thresh_str, current_lr],
            header,
        )

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            epochs_no_improve = 0
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model for {model_name}...")
            state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "model_name": model_name,
                },
                best_model_path,
            )

        state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "model_name": model_name,
            },
            last_model_path,
        )
        print(f"Checkpoint saved to {last_model_path}")

        if not improved:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    starting_epoch = 0

    t_end_training = time.time()
    print(f"Training finished. Total time: {t_end_training - t_start_training:.2f} s")
    writer.flush()
    writer.close()

    # Load best model for final evaluation/plots
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        if use_dataparallel:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    _, _, _, y_true, y_prob, y_pred, best_thresh = _run_epoch(
        model,
        val_loader,
        val_criterion,
        optimizer=None,
        device=device,
        phase="val",
        threshold=threshold,
        auto_threshold=True,
        abnormal_model=abnormal_model,
        apply_abnormal_prior=apply_abnormal_prior,
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{model_name}_{task_name}_curves.png"))
    _plot_confusion_matrix(y_true, y_pred, os.path.join(eval_folder, f"{model_name}_{task_name}_confusion.png"))
    _plot_roc(y_true, y_prob, os.path.join(eval_folder, f"{model_name}_{task_name}_roc.png"))

    metrics_summary = _compute_confusion_metrics(y_true, y_pred)
    metrics_summary["best_threshold"] = float(best_thresh)
    summary_path = os.path.join(eval_folder, f"{model_name}_{task_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in metrics_summary.items():
            f.write(f"{k}: {v}\n")

    print(
        "Summary | Acc {:.4f} | Sens {:.4f} | Spec {:.4f} | Prec {:.4f} | F1 {:.4f}".format(
            metrics_summary["accuracy"],
            metrics_summary["sensitivity"],
            metrics_summary["specificity"],
            metrics_summary["precision"],
            metrics_summary["f1"],
        )
    )

    print(f"Metrics saved to: {csv_path}")
    print(f"Plots saved to: {eval_folder}")
    print(f"Summary saved to: {summary_path}")

    val_best_thresh = best_thresh
    if run_test:
        test_ids, test_map = _read_split_labels(labels_dir, task_name, "test")
        if test_ids:
            test_loader = _load_test_from_ids(
                test_ids,
                test_map,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                target_slices=config["target_slices"],
                image_size=config["image_size"],
                data_dir=data_dir,
            )
            _, _, _, y_true_t, y_prob_t, y_pred_t, best_thresh_t = _run_epoch(
                model,
                test_loader,
                val_criterion,
                optimizer=None,
                device=device,
                phase="test",
                threshold=val_best_thresh,
                auto_threshold=False,
                abnormal_model=abnormal_model,
                apply_abnormal_prior=apply_abnormal_prior,
            )
            _plot_confusion_matrix(
                y_true_t, y_pred_t, os.path.join(eval_folder, f"{model_name}_{task_name}_test_confusion.png")
            )
            _plot_roc(
                y_true_t, y_prob_t, os.path.join(eval_folder, f"{model_name}_{task_name}_test_roc.png")
            )
            test_summary = _compute_confusion_metrics(y_true_t, y_pred_t)
            test_summary["best_threshold"] = float(best_thresh_t)
            test_path = os.path.join(eval_folder, f"{model_name}_{task_name}_test_summary.txt")
            with open(test_path, "w", encoding="utf-8") as f:
                for k, v in test_summary.items():
                    f.write(f"{k}: {v}\n")
            print(f"Test summary saved to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetb0",
        choices=["densenet121", "efficientnetb0", "efficientnetb0_se_vit"],
        help="Choose model to train",
    )
    parser.add_argument("--labels-dir", type=str, default="labels", help="Path to labels directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting prob to class")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto/cpu/cuda",
    )
    parser.add_argument("--abnormal-model", type=str, default=None, help="Path to abnormality prior model")
    args = parser.parse_args()

    cfg = dict(base_config)
    task = cfg.get("task", "acl")
    print("Training Configuration")
    print(cfg)
    train(
        config=cfg,
        model_name=args.model,
        threshold=args.threshold,
        labels_dir=args.labels_dir,
        data_dir=args.data_dir,
        run_test=True,
        device_arg=args.device,
        abnormal_model_path=args.abnormal_model,
    )
    print("Training Ended...")
