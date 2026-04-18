from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int32)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float32)
    y_pred_arr = (y_prob_arr >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "loss": float("nan"),
        "auc": float("nan"),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    if len(np.unique(y_true_arr)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        out["auc"] = float(auc(fpr, tpr))

    return out


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    matrix_plot = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(matrix_plot, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_curve(y_true: Iterable[int], y_prob: Iterable[float], out_path: Path, title: str) -> bool:
    y_true_arr = np.asarray(list(y_true), dtype=np.int32)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float32)
    if len(np.unique(y_true_arr)) < 2:
        return False

    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    roc_auc = auc(fpr, tpr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def save_training_curves(history_rows: List[Dict[str, float]], out_path: Path) -> None:
    if not history_rows:
        return

    epochs = [int(row["epoch"]) for row in history_rows]
    train_loss = [float(row["train_loss"]) for row in history_rows]
    val_loss = [float(row["val_loss"]) for row in history_rows]
    train_auc = [float(row["train_auc"]) for row in history_rows]
    val_auc = [float(row["val_auc"]) for row in history_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, train_auc, label="Train AUC", marker="o")
    axes[1].plot(epochs, val_auc, label="Val AUC", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC Curve")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
