import argparse
import json
import numpy as np
import os
import re
import torch

from datetime import datetime
from pathlib import Path

from evaluate import run_model, save_confusion_matrix, save_roc_curve
from loader import load_data
from model import TripleMRNet


def _find_latest_model(rundir):
    """
    Quét rundir tìm file model epoch cao nhất (không phải checkpoint/ và không phải .json).
    Trả về (max_epoch, model_path) hoặc (0, None).
    """
    max_epoch  = 0
    model_path = None
    for dirpath, _, files in os.walk(rundir):
        # Bỏ qua thư mục checkpoints để không bị nhầm
        if "checkpoints" in dirpath:
            continue
        for fname in files:
            if fname.endswith(".json") or "checkpoint" in fname:
                continue
            match = re.search(r"epoch(\d+)", fname)
            if not match:
                continue
            ep = int(match.group(1))
            if ep >= max_epoch:
                max_epoch  = ep
                model_path = os.path.join(dirpath, fname)
    return max_epoch, model_path


def _find_latest_checkpoint(checkpoint_dir):
    """
    Quét checkpoint_dir tìm checkpoint epoch cao nhất.
    Trả về (max_epoch, ckpt_path) hoặc (0, None).
    """
    max_epoch = 0
    ckpt_path = None
    if not checkpoint_dir.exists():
        return 0, None
    for fname in os.listdir(checkpoint_dir):
        match = re.search(r"epoch(\d+)", fname)
        if not match:
            continue
        ep = int(match.group(1))
        if ep >= max_epoch:
            max_epoch = ep
            ckpt_path = checkpoint_dir / fname
    return max_epoch, ckpt_path


def train(
    rundir,
    task,
    backbone,
    epochs,
    learning_rate,
    weight_decay,           # ← nhận tường minh, không dùng args global
    use_gpu,
    abnormal_model_path=None,
    data_dir="data",
    labels_dir="labels",
    num_workers=4,
    use_amp=False,
    checkpoint_every=1,
):
    train_loader, valid_loader, test_loader = load_data(
        task, use_gpu,
        data_dir=data_dir,
        labels_dir=labels_dir,
        num_workers=num_workers,
        backbone=backbone,
    )

    if abnormal_model_path and not os.path.isfile(abnormal_model_path):
        print(
            f"[Warning] abnormal_model_path not found: {abnormal_model_path}. "
            "Disable abnormal gate and continue training/evaluation."
        )
        abnormal_model_path = None

    model = TripleMRNet(backbone=backbone)
    if use_gpu:
        model = model.cuda()

    # Optimizer và Scheduler (tạo trước khi load checkpoint để có thể restore state)
    backbone_params   = (list(model.axial_net.parameters()) +
                         list(model.sagit_net.parameters()) +
                         list(model.coron_net.parameters()))
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,   'lr': learning_rate * 0.1},
        {'params': classifier_params, 'lr': learning_rate},
    ], weight_decay=weight_decay)   # ← dùng tham số đã truyền vào, không phải args

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, threshold=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume: ưu tiên checkpoint (có optimizer state) hơn best-model ──────
    start_epoch = 0
    checkpoint_dir = Path(rundir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_epoch, ckpt_path = _find_latest_checkpoint(checkpoint_dir)
    if ckpt_path:
        print(f"[Resume] Load checkpoint epoch {ckpt_epoch}: {ckpt_path}")
        ckpt = torch.load(ckpt_path,
                          map_location=(None if use_gpu else 'cpu'))
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"]
    else:
        # Fallback: chỉ có file model (không có optimizer state)
        model_epoch, model_path = _find_latest_model(rundir)
        if model_path:
            print(f"[Resume] Load model-only epoch {model_epoch}: {model_path}")
            state = torch.load(model_path,
                               map_location=(None if use_gpu else 'cpu'))
            model.load_state_dict(state)
            start_epoch = model_epoch

    # ── Logging ──────────────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    best_model_path = Path(rundir) / "best_model.pth"

    log_path = Path(rundir) / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    start_time = datetime.now()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        elapsed = datetime.now() - start_time
        print(f"Starting epoch {epoch + 1}. Time passed: {elapsed}")

        train_loss, train_auc, _, _ = run_model(
            model, train_loader,
            train=True, optimizer=optimizer,
            abnormal_model_path=abnormal_model_path,
            use_amp=use_amp, scaler=scaler,
        )
        print(f"  train loss: {train_loss:.4f}  train AUC: {train_auc:.4f}")
        if use_gpu:
            torch.cuda.empty_cache()

        val_loss, val_auc, _, _ = run_model(
            model, valid_loader,
            abnormal_model_path=abnormal_model_path,
            use_amp=use_amp,
        )
        print(f"  val   loss: {val_loss:.4f}  val   AUC: {val_auc:.4f}")

        scheduler.step(val_loss)

        # Log LR của classifier group (index 1)
        lr = optimizer.param_groups[1]["lr"]
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_auc:.6f},"
                    f"{val_loss:.6f},{val_auc:.6f},{lr:.8f}\n")

        # Checkpoint đầy đủ (model + optimizer + scaler)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_save = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state":    scaler.state_dict(),
                "val_loss":        val_loss,
                "val_auc":         val_auc,
            }, ckpt_save)

        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_name = f"val{val_loss:.4f}_train{train_loss:.4f}_epoch{epoch+1}"
            torch.save(model.state_dict(), Path(rundir) / file_name)
            torch.save(model.state_dict(), best_model_path)

    # ── Evaluate trên test set ────────────────────────────────────────────────
    if best_model_path.exists():
        best_state = torch.load(best_model_path,
                                map_location=(None if use_gpu else 'cpu'))
        model.load_state_dict(best_state)
    model.eval()
    if use_gpu:
        torch.cuda.empty_cache()

    test_loss, test_auc, preds, labels = run_model(
        model, test_loader,
        abnormal_model_path=abnormal_model_path,
        use_amp=use_amp,
    )
    print(f"test loss: {test_loss:.4f}  test AUC: {test_auc:.4f}")

    with open(log_path, "a") as f:
        f.write(f"test,{test_loss:.6f},{test_auc:.6f},,,\n")

    save_confusion_matrix(labels, preds, Path(rundir) / "confusion_matrix.png")
    save_roc_curve(labels, preds, Path(rundir) / "roc_curve.png")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir',      type=str,   required=True)
    parser.add_argument('--task',        type=str,   required=True)
    parser.add_argument('--data-dir',    type=str,   default="data")
    parser.add_argument('--labels-dir',  type=str,   default="labels")
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--gpu',         action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--backbone',    type=str,   default="efficientnet_b0")
    parser.add_argument('--abnormal_model', type=str, default=None)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--amp',         action='store_true')
    parser.add_argument('--checkpoint_every', type=int, default=1)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.task != "abnormal" and args.abnormal_model is None:
        raise ValueError(
            "Cần cung cấp --abnormal_model cho task 'acl' hoặc 'meniscus'")

    os.makedirs(args.rundir, exist_ok=True)
    with open(Path(args.rundir) / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    train(
        rundir=args.rundir,
        task=args.task,
        backbone=args.backbone,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,     # ← truyền tường minh vào hàm
        use_gpu=args.gpu,
        abnormal_model_path=args.abnormal_model,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        num_workers=args.num_workers,
        use_amp=(args.gpu and args.amp),
        checkpoint_every=args.checkpoint_every,
    )
