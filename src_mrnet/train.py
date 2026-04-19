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
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    is_train: bool,
    log_interval: int,
    logger: logging.Logger,
) -> Tuple[float, List[int], List[float]]:
    mode = "Train" if is_train else "Valid"
    model.train(is_train)

    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    progress = tqdm(
        loader,
        desc=f"{mode} {epoch_idx + 1}/{total_epochs}",
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
            logits = model(axial_batch, sagittal_batch, coronal_batch)
            loss = criterion(logits, labels)
            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits.detach()).flatten().cpu().tolist()
        targets = labels.detach().flatten().cpu().int().tolist()

        losses.append(float(loss.item()))
        all_probs.extend(float(prob) for prob in probs)
        all_labels.extend(int(target) for target in targets)

        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            avg_loss = float(np.mean(losses))
            progress.set_postfix(loss=f"{avg_loss:.4f}")
            logger.info(
                "[%s] Epoch %d/%d | Step %d/%d | Loss %.4f",
                mode,
                epoch_idx + 1,
                total_epochs,
                step,
                len(loader),
                avg_loss,
            )

    epoch_loss = float(np.mean(losses)) if losses else float("inf")
    return epoch_loss, all_labels, all_probs


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


def _fmt_metric(value: float) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _log_table_header(logger: logging.Logger) -> None:
    border = "+-------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+"
    header = "| Epoch | TrainLoss | TrainAUC  | TrainAcc  | ValLoss   | ValAUC    | ValAcc    | ValF1     | LR       |"
    logger.info(border)
    logger.info(header)
    logger.info(border)


def _log_table_row(
    logger: logging.Logger,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    learning_rate: float,
) -> None:
    row = (
        f"| {epoch:5d} | "
        f"{_fmt_metric(train_metrics['loss']):>9} | "
        f"{_fmt_metric(train_metrics['auc']):>9} | "
        f"{_fmt_metric(train_metrics['accuracy']):>9} | "
        f"{_fmt_metric(val_metrics['loss']):>9} | "
        f"{_fmt_metric(val_metrics['auc']):>9} | "
        f"{_fmt_metric(val_metrics['accuracy']):>9} | "
        f"{_fmt_metric(val_metrics['f1']):>9} | "
        f"{learning_rate:>8.2e} |"
    )
    logger.info(row)


def train(config: TrainConfig) -> None:
    run_dir = Path("runs") / config.exp_name
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(run_dir=run_dir)
    device = make_device(config)
    logger.info("Using device: %s", device)

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.target_slices <= 0:
        raise ValueError("target_slices must be > 0.")
    if config.max_epoch <= 0:
        raise ValueError("max_epoch must be > 0.")

    set_seed(config.seed)
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2, ensure_ascii=False)

    train_loader, valid_loader = create_loaders(
        task=config.task,
        data_dir=config.data_dir,
        labels_dir=config.labels_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        target_slices=config.target_slices,
        image_size=config.image_size,
    )
    logger.info("Dataset loaded: train=%d | valid=%d", len(train_loader.dataset), len(valid_loader.dataset))

    model = TripleMRNetEfficientNetB0(pretrained=config.pretrained).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )

    pos_weight = compute_pos_weight(train_loader.dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history_rows: List[Dict[str, float]] = []
    history_path = run_dir / "history.csv"
    best_model_path = run_dir / "best_model.pth"
    best_metrics_path = run_dir / "best_metrics.json"
    training_curves_path = figures_dir / "training_curves.png"
    best_cm_path = figures_dir / "best_confusion_matrix.png"
    best_roc_path = figures_dir / "best_roc_curve.png"

    best_score = -float("inf")
    best_epoch = 0
    no_improve_count = 0
    started_at = time.time()

    logger.info("Start training for %d epochs (from epoch %d).", config.max_epoch, config.starting_epoch)
    _log_table_header(logger)

    for epoch in range(config.starting_epoch, config.max_epoch):
        train_loss, train_y_true, train_y_prob = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=config.max_epoch,
            is_train=True,
            log_interval=config.log_train,
            logger=logger,
        )
        train_metrics = compute_binary_metrics(train_y_true, train_y_prob, threshold=config.threshold)
        train_metrics["loss"] = train_loss

        val_loss, val_y_true, val_y_prob = run_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch_idx=epoch,
            total_epochs=config.max_epoch,
            is_train=False,
            log_interval=config.log_val,
            logger=logger,
        )
        val_metrics = compute_binary_metrics(val_y_true, val_y_prob, threshold=config.threshold)
        val_metrics["loss"] = val_loss

        score_for_sched = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -val_loss
        scheduler.step(score_for_sched)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_num = epoch + 1
        epoch_row = {
            "epoch": epoch_num,
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
        save_training_curves(history_rows, training_curves_path)
        _log_table_row(logger, epoch=epoch_num, train_metrics=train_metrics, val_metrics=val_metrics, learning_rate=current_lr)

        checkpoint_payload = {
            "epoch": epoch_num,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config_to_dict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint_payload, checkpoints_dir / "last_checkpoint.pth")
        logger.info("Saved: %s", (checkpoints_dir / "last_checkpoint.pth").as_posix())
        if config.save_model > 0 and epoch_num % config.save_model == 0:
            torch.save(checkpoint_payload, checkpoints_dir / f"checkpoint_epoch_{epoch_num:03d}.pth")
            logger.info("Saved: %s", (checkpoints_dir / f"checkpoint_epoch_{epoch_num:03d}.pth").as_posix())

        current_score = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -val_metrics["loss"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch_num
            no_improve_count = 0
            torch.save(checkpoint_payload, best_model_path)
            logger.info("Saved: %s", best_model_path.as_posix())

            best_metrics = {
                "epoch": epoch_num,
                "score": float(current_score),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            with open(best_metrics_path, "w", encoding="utf-8") as file:
                json.dump(best_metrics, file, indent=2, ensure_ascii=False)

            cm = np.array(
                [
                    [int(val_metrics["tn"]), int(val_metrics["fp"])],
                    [int(val_metrics["fn"]), int(val_metrics["tp"])],
                ],
                dtype=np.int32,
            )
            save_confusion_matrix(cm, best_cm_path, title=f"Best Confusion Matrix (Epoch {epoch_num})")
            roc_saved = save_roc_curve(
                y_true=val_y_true,
                y_prob=val_y_prob,
                out_path=best_roc_path,
                title=f"Best ROC Curve (Epoch {epoch_num})",
            )
            if roc_saved:
                logger.info("Saved: %s", best_roc_path.as_posix())
            else:
                logger.info("Skipped ROC curve (validation labels have only one class).")
            logger.info("Saved: %s", best_cm_path.as_posix())
            logger.info("Saved: %s", best_metrics_path.as_posix())
            logger.info("Best model updated at epoch %d (score=%.4f).", epoch_num, best_score)
        else:
            no_improve_count += 1

        if config.patience > 0 and no_improve_count >= config.patience:
            logger.info("Early stopping at epoch %d (no improvement %d epochs).", epoch_num, no_improve_count)
            break

    logger.info(
        "Training finished in %.1f min | best score=%.4f | best epoch=%d | artifacts=%s",
        (time.time() - started_at) / 60.0,
        best_score,
        best_epoch,
        run_dir.resolve(),
    )


if __name__ == "__main__":
    cfg = build_config()
    train(cfg)
