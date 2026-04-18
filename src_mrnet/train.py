import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src_mrnet.config import TrainConfig, build_config, config_to_dict
from src_mrnet.loader import compute_pos_weight, create_loaders
from src_mrnet.metrics import (
    compute_binary_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_training_curves,
)
from src_mrnet.model import TripleMRNetEfficientNetB0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mrnet_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def make_device(config: TrainConfig) -> torch.device:
    use_cuda = config.use_gpu and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def move_volume_batch_to_device(volume_batch: Iterable[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [vol.to(device, non_blocking=True) for vol in volume_batch]


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    is_train: bool,
    mixed_precision: bool,
    grad_clip_norm: float,
    log_interval: int,
) -> Tuple[float, List[int], List[float]]:
    mode = "Train" if is_train else "Valid"
    model.train(is_train)

    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    progress = tqdm(
        loader,
        desc=f"{mode} {epoch_idx}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, (vol_axial, vol_sagittal, vol_coronal, labels, _) in enumerate(progress, start=1):
        axial_batch = move_volume_batch_to_device(vol_axial, device)
        sagittal_batch = move_volume_batch_to_device(vol_sagittal, device)
        coronal_batch = move_volume_batch_to_device(vol_coronal, device)
        labels = labels.to(device, non_blocking=True)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                logits = model(axial_batch, sagittal_batch, coronal_batch)
                loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                if scaler is not None and mixed_precision:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

        probs = torch.sigmoid(logits.detach()).flatten().cpu().tolist()
        targets = labels.detach().flatten().cpu().int().tolist()

        losses.append(float(loss.item()))
        all_probs.extend(float(prob) for prob in probs)
        all_labels.extend(int(target) for target in targets)

        if step % max(log_interval, 1) == 0 or step == len(loader):
            avg_loss = float(np.mean(losses))
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

    epoch_loss = float(np.mean(losses)) if losses else float("inf")
    return epoch_loss, all_labels, all_probs


def split_optimizer_parameters(model: TripleMRNetEfficientNetB0):
    backbone_params = []
    for encoder in (model.axial_encoder, model.sagittal_encoder, model.coronal_encoder):
        backbone_params.extend(list(encoder.parameters()))
    classifier_params = list(model.classifier.parameters())
    return backbone_params, classifier_params


def write_history_csv(history_path: Path, history_rows: List[Dict[str, float]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_accuracy",
        "val_loss",
        "val_auc",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "learning_rate",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def train(config: TrainConfig) -> None:
    run_dir = Path(config.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(run_dir=run_dir)
    device = make_device(config)
    logger.info("Using device: %s", device)

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.image_size <= 0:
        raise ValueError("image_size must be > 0.")

    set_seed(config.seed)
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2, ensure_ascii=False)

    train_loader, valid_loader = create_loaders(
        task=config.task,
        data_dir=config.data_dir,
        labels_dir=config.labels_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("Dataset loaded: train=%d | valid=%d", len(train_loader.dataset), len(valid_loader.dataset))

    model = TripleMRNetEfficientNetB0(
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    backbone_params, classifier_params = split_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": config.learning_rate * config.backbone_lr_mult,
            },
            {
                "params": classifier_params,
                "lr": config.learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )

    pos_weight = compute_pos_weight(train_loader.dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision and device.type == "cuda"))

    history_rows: List[Dict[str, float]] = []
    history_path = run_dir / "history.csv"
    best_model_path = run_dir / "best_model.pth"
    best_metrics_path = run_dir / "best_metrics.json"

    best_score = -float("inf")
    best_epoch = -1
    best_wait = 0
    started_at = time.time()

    logger.info("Start training for %d epochs.", config.epochs)
    for epoch in range(1, config.epochs + 1):
        train_loss, train_y_true, train_y_prob = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch_idx=epoch,
            total_epochs=config.epochs,
            is_train=True,
            mixed_precision=config.mixed_precision and device.type == "cuda",
            grad_clip_norm=config.grad_clip_norm,
            log_interval=config.log_interval,
        )
        train_metrics = compute_binary_metrics(train_y_true, train_y_prob, threshold=config.threshold)
        train_metrics["loss"] = train_loss

        val_loss, val_y_true, val_y_prob = run_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            epoch_idx=epoch,
            total_epochs=config.epochs,
            is_train=False,
            mixed_precision=False,
            grad_clip_norm=config.grad_clip_norm,
            log_interval=config.log_interval,
        )
        val_metrics = compute_binary_metrics(val_y_true, val_y_prob, threshold=config.threshold)
        val_metrics["loss"] = val_loss

        score_for_sched = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -val_loss
        scheduler.step(score_for_sched)
        current_lr = optimizer.param_groups[-1]["lr"]

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "learning_rate": current_lr,
        }
        history_rows.append(epoch_row)
        write_history_csv(history_path, history_rows)

        elapsed_min = (time.time() - started_at) / 60.0
        logger.info(
            (
                "[Epoch %d/%d] "
                "train_loss=%.4f train_auc=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_auc=%.4f val_acc=%.4f val_f1=%.4f | "
                "lr=%.2e elapsed=%.1f min"
            ),
            epoch,
            config.epochs,
            train_metrics["loss"],
            train_metrics["auc"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            current_lr,
            elapsed_min,
        )

        latest_ckpt = checkpoints_dir / "last_checkpoint.pth"
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "config": config_to_dict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint_payload, latest_ckpt)

        if config.save_every > 0 and epoch % config.save_every == 0:
            torch.save(checkpoint_payload, checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth")

        current_score = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -val_metrics["loss"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_wait = 0

            torch.save(checkpoint_payload, best_model_path)
            with open(best_metrics_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "val_metrics": val_metrics,
                        "threshold": config.threshold,
                    },
                    file,
                    indent=2,
                )

            cm = np.array(
                [
                    [int(val_metrics["tn"]), int(val_metrics["fp"])],
                    [int(val_metrics["fn"]), int(val_metrics["tp"])],
                ],
                dtype=np.int32,
            )
            save_confusion_matrix(
                cm=cm,
                out_path=figures_dir / "best_confusion_matrix.png",
                title=f"Validation Confusion Matrix (Epoch {epoch})",
            )
            roc_saved = save_roc_curve(
                y_true=val_y_true,
                y_prob=val_y_prob,
                out_path=figures_dir / "best_roc_curve.png",
                title=f"Validation ROC (Epoch {epoch})",
            )
            if not roc_saved:
                logger.warning("ROC curve skipped at epoch %d because validation has only one class.", epoch)
            logger.info("Best model updated at epoch %d. Saved to %s", epoch, best_model_path)
        else:
            best_wait += 1

        if config.early_stopping_patience > 0 and best_wait >= config.early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs).",
                epoch,
                best_wait,
            )
            break

    save_training_curves(history_rows=history_rows, out_path=figures_dir / "training_curves.png")
    logger.info("Training finished. Best epoch=%d | best score=%.4f", best_epoch, best_score)
    logger.info("Artifacts: %s", run_dir.resolve())


if __name__ == "__main__":
    cfg = build_config()
    train(cfg)
