import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from dataset import load_data
from dataset.dataset import INPUT_DIM, MAX_PIXEL_VAL, MEAN, STDDEV
from preprocessing.slice_sampling import uniform_slice_sampling
from preprocessing.augmentation import random_augmentation
from config import config as base_config
from models import Densenet121, EfficientNetB0
from utils import _get_lr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _build_model(name: str):
    name = name.lower()
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
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

        self.labels = [self.labels_map[rid] for rid in self.ids]

        pos = sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([neg / pos]) if pos > 0 else torch.FloatTensor([1.0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            if self.target_slices is not None:
                img_raw[plane] = uniform_slice_sampling(img_raw[plane], self.target_slices)
            if self.train:
                vol = torch.from_numpy(img_raw[plane])
                vol = random_augmentation(vol)
                img_raw[plane] = vol.numpy()
            img_raw[plane] = self._resize_image(img_raw[plane])

        label = self.labels[index]
        label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])
        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        target = self.input_dim
        if target is not None and target <= image.shape[1] and target <= image.shape[2]:
            pad = int((image.shape[2] - target) / 2)
            image = image[:, pad:-pad, pad:-pad]

        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * MAX_PIXEL_VAL
        image = (image - MEAN) / STDDEV
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


def _read_task_labels(labels_dir, task):
    labels_map = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(labels_dir, f"{split}-{task}.csv")
        data = np.genfromtxt(path, delimiter=",", dtype=int)
        if data.ndim == 1 and data.size == 2:
            data = np.array([data])
        for rid, lab in data:
            labels_map[int(rid)] = int(lab)
    ids = sorted(labels_map.keys())
    return ids, labels_map


def _load_data_from_ids(ids_train, ids_val, labels_map, batch_size, num_workers, target_slices, image_size, data_dir):
    augments = transforms.Compose(
        [
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)),
            transforms.RandomHorizontalFlip(),
        ]
    )

    train_data = MRDataByIds(
        ids_train,
        labels_map,
        data_dir=data_dir,
        train=True,
        transform=augments,
        target_slices=target_slices,
        input_dim=image_size,
    )
    labels = train_data.labels
    class_counts = np.bincount(labels) if len(labels) > 0 else np.array([1, 1])
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    val_data = MRDataByIds(
        ids_val,
        labels_map,
        data_dir=data_dir,
        train=False,
        transform=None,
        target_slices=target_slices,
        input_dim=image_size,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, train_data.weights, val_data.weights


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu", phase="train"):
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
            output = model(images)
            loss = criterion(output, label)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())

        probas = torch.sigmoid(output).detach().cpu().view(-1).numpy().tolist()
        labels = label.detach().cpu().view(-1).numpy().tolist()

        y_prob.extend(probas)
        y_true.extend(labels)

    if len(losses) == 0:
        return 0.0, 0.5, 0.0, [], [], []

    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = metrics.accuracy_score(y_true, y_pred)
    loss_mean = float(np.mean(losses))
    return loss_mean, float(auc), float(acc), y_true, y_prob, y_pred


def _append_csv(csv_path, row, header):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _plot_curves(csv_path, out_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    epochs = np.atleast_1d(data["epoch"])
    train_loss = np.atleast_1d(data["train_loss"])
    val_loss = np.atleast_1d(data["val_loss"])
    train_auc = np.atleast_1d(data["train_auc"])
    val_auc = np.atleast_1d(data["val_auc"])
    train_acc = np.atleast_1d(data["train_acc"])
    val_acc = np.atleast_1d(data["val_acc"])

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


def train(config: dict, model_name: str, loaders=None, fold_idx=None, resume=True):
    fold_suffix = f"fold_{fold_idx}" if fold_idx is not None else None
    save_folder = os.path.join("weights", config["task"])
    if fold_suffix:
        save_folder = os.path.join(save_folder, fold_suffix)
    os.makedirs(save_folder, exist_ok=True)

    eval_folder = os.path.join("evaluation", f"{model_name}_{config['task']}")
    if fold_suffix:
        eval_folder = os.path.join(eval_folder, fold_suffix)
    os.makedirs(eval_folder, exist_ok=True)

    csv_path = os.path.join(eval_folder, f"{model_name}_{config['task']}_metrics.csv")
    best_model_path = os.path.join(save_folder, f"{model_name}_best_model.pth")
    last_model_path = os.path.join(save_folder, f"{model_name}_last_checkpoint.pth")

    print("Starting to Train Model...")
    if loaders is None:
        train_loader, val_loader, train_wts, val_wts = load_data(
            config["task"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            target_slices=config["target_slices"],
            image_size=config["image_size"],
        )
    else:
        train_loader, val_loader, train_wts, val_wts = loaders

    print("Initializing Model...")
    model = _build_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()

    print("Initializing Loss Method...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)
    if device == "cuda":
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print("Setup the Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.3, threshold=1e-4
    )

    starting_epoch = config["starting_epoch"]
    num_epochs = config["max_epoch"]
    best_val_auc = float(0)
    patience = config.get("patience", 5)
    epochs_no_improve = 0

    if resume and os.path.exists(last_model_path):
        print(f"Found checkpoint at {last_model_path}. Loading...")
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        starting_epoch = checkpoint.get("epoch", starting_epoch) + 1
        best_val_auc = checkpoint.get("best_val_auc", best_val_auc)
        print(f"Resuming from epoch {starting_epoch} | Best AUC {best_val_auc:.4f}")

    writer = SummaryWriter(comment=f"model={model_name} lr={config['lr']} task={config['task']} fold={fold_idx}")
    t_start_training = time.time()

    header = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_acc",
        "val_loss",
        "val_auc",
        "val_acc",
        "lr",
    ]

    for epoch in range(starting_epoch, num_epochs):
        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        train_loss, train_auc, train_acc, _, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device, phase="train"
        )
        val_loss, val_auc, val_acc, _, _, _ = _run_epoch(
            model, val_loader, val_criterion, optimizer=None, device=device, phase="val"
        )

        writer.add_scalar("Train/Avg Loss", train_loss, epoch)
        writer.add_scalar("Train/AUC_epoch", train_auc, epoch)
        writer.add_scalar("Train/Acc_epoch", train_acc, epoch)
        writer.add_scalar("Val/Avg Loss", val_loss, epoch)
        writer.add_scalar("Val/AUC_epoch", val_auc, epoch)
        writer.add_scalar("Val/Acc_epoch", val_acc, epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time
        print(
            "Epoch [{}/{}] | train loss {:.4f} | train auc {:.4f} | train acc {:.4f} | "
            "val loss {:.4f} | val auc {:.4f} | val acc {:.4f} | time {:.2f} s".format(
                epoch, num_epochs, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, delta
            )
        )
        print("-" * 30)
        writer.flush()

        _append_csv(
            csv_path,
            [epoch, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, current_lr],
            header,
        )

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            epochs_no_improve = 0
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model for {model_name}...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "model_name": model_name,
                },
                best_model_path,
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
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

    t_end_training = time.time()
    print(f"Training finished. Total time: {t_end_training - t_start_training:.2f} s")
    writer.flush()
    writer.close()

    # Load best model for final evaluation/plots
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    _, _, _, y_true, y_prob, y_pred = _run_epoch(
        model, val_loader, val_criterion, optimizer=None, device=device, phase="val"
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{model_name}_{config['task']}_curves.png"))
    _plot_confusion_matrix(y_true, y_pred, os.path.join(eval_folder, f"{model_name}_{config['task']}_confusion.png"))
    _plot_roc(y_true, y_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_roc.png"))

    metrics_summary = _compute_confusion_metrics(y_true, y_pred)
    summary_path = os.path.join(eval_folder, f"{model_name}_{config['task']}_summary.txt")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetb0",
        choices=["densenet121", "efficientnetb0"],
        help="Choose model to train",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated tasks to train (default: task in config.py)",
    )
    parser.add_argument("--kfold", type=int, default=0, help="Enable K-Fold if > 1 (e.g., 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for K-Fold split")
    parser.add_argument("--labels-dir", type=str, default="labels", help="Path to labels directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        tasks = [base_config["task"]]
    for task in tasks:
        cfg = dict(base_config)
        cfg["task"] = task
        print("Training Configuration")
        print(cfg)
        if args.kfold and args.kfold > 1:
            ids, labels_map = _read_task_labels(args.labels_dir, task)
            y = [labels_map[i] for i in ids]
            skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
            fold_aucs = []
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, y)):
                ids_train = [ids[i] for i in train_idx]
                ids_val = [ids[i] for i in val_idx]
                loaders = _load_data_from_ids(
                    ids_train,
                    ids_val,
                    labels_map,
                    batch_size=cfg["batch_size"],
                    num_workers=cfg["num_workers"],
                    target_slices=cfg["target_slices"],
                    image_size=cfg["image_size"],
                    data_dir=args.data_dir,
                )
                print(f"=== Task {task} | Fold {fold_idx + 1}/{args.kfold} ===")
                train(config=cfg, model_name=args.model, loaders=loaders, fold_idx=fold_idx, resume=False)
                # Read best AUC from checkpoint after fold
                best_path = os.path.join("weights", task, f"fold_{fold_idx}", f"{args.model}_best_model.pth")
                if os.path.exists(best_path):
                    checkpoint = torch.load(best_path, map_location="cpu")
                    fold_aucs.append(float(checkpoint.get("best_val_auc", 0.0)))
            if fold_aucs:
                mean_auc = float(np.mean(fold_aucs))
                std_auc = float(np.std(fold_aucs))
                print(f"K-Fold Summary | Task {task} | AUC {mean_auc:.4f} ± {std_auc:.4f}")
        else:
            train(config=cfg, model_name=args.model)
    print("Training Ended...")
