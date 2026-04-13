import argparse
import json
import numpy as np
import os
import torch
import re

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model, save_confusion_matrix, save_roc_curve
from loader import load_data
from model import TripleMRNet

def train(
    rundir,
    task,
    backbone,
    epochs,
    learning_rate,
    use_gpu,
    abnormal_model_path=None,
    data_dir="data",
    labels_dir="labels",
    num_workers=8,
    use_amp=False,
    checkpoint_every=1,
):
    train_loader, valid_loader, test_loader = load_data(
        task,
        use_gpu,
        data_dir=data_dir,
        labels_dir=labels_dir,
        num_workers=num_workers,
        backbone=backbone,
    )

    if abnormal_model_path and not os.path.isfile(abnormal_model_path):
        raise FileNotFoundError(f"abnormal_model_path not found: {abnormal_model_path}")
    
    model = TripleMRNet(backbone=backbone)
    max_epoch = 0
    for dirpath, dirnames, files in os.walk(args.rundir):
        if not files:
            break
        model_path = None
        for fname in files:
            if fname.endswith(".json"):
                continue
            if "checkpoint" in fname:
                continue
            match = re.search(r"epoch(\d+)", fname)
            if not match:
                continue
            ep = int(match.group(1))
            if ep >= max_epoch:
                max_epoch = ep
                model_path = os.path.join(dirpath, fname)
        
        if model_path:
            state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
            model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_model_path = Path(rundir) / "best_model.pth"
    checkpoint_dir = Path(rundir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(rundir) / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as logf:
            logf.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    start_time = datetime.now()

    epoch = 0
    if max_epoch: epoch += max_epoch
    while epoch < epochs:
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(
                model, train_loader, train=True, optimizer=optimizer,
                abnormal_model_path=abnormal_model_path, use_amp=use_amp, scaler=scaler)

        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader,
                abnormal_model_path=abnormal_model_path, use_amp=use_amp)
        
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        scheduler.step(val_loss)

        with open(log_path, "a") as logf:
            lr = optimizer.param_groups[0]["lr"]
            logf.write(f"{epoch+1},{train_loss:.6f},{train_auc:.6f},{val_loss:.6f},{val_auc:.6f},{lr:.8f}\n")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                },
                ckpt_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), best_model_path)

        epoch += 1

    # Evaluate best model on test set and save plots
    if best_model_path.exists():
        best_state = torch.load(best_model_path, map_location=(None if use_gpu else 'cpu'))
        model.load_state_dict(best_state)
    model.eval()
    test_loss, test_auc, preds, labels = run_model(
        model, test_loader, abnormal_model_path=abnormal_model_path, use_amp=use_amp
    )
    print(f'test loss: {test_loss:0.4f}')
    print(f'test AUC: {test_auc:0.4f}')
    with open(log_path, "a") as logf:
        logf.write(f"test,{test_loss:.6f},{test_auc:.6f},,,\n")

    save_confusion_matrix(labels, preds, Path(rundir) / "confusion_matrix.png")
    save_roc_curve(labels, preds, Path(rundir) / "roc_curve.png")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--labels-dir', type=str, default="labels")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--backbone', default="efficientnet_b0", type=str)
    parser.add_argument('--abnormal_model', default=None, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--checkpoint_every', default=1, type=int)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.task != "abnormal":
        if args.abnormal_model is None:
            raise ValueError("Enter abnormal model path for `acl` or `meniscus` task")

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(
        args.rundir,
        args.task,
        args.backbone,
        args.epochs,
        args.learning_rate,
        args.gpu,
        abnormal_model_path=args.abnormal_model,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        num_workers=args.num_workers,
        use_amp=(args.gpu or args.amp),
        checkpoint_every=args.checkpoint_every,
    )
