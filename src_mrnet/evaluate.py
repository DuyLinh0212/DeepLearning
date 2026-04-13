import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from tqdm import tqdm

import pdb

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import TripleMRNet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def run_model(model, loader, train=False, optimizer=None,
        abnormal_model_path=None, use_amp=False, scaler=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        if abnormal_model_path:
            abnormal_model = TripleMRNet(backbone=model.backbone)
            state_dict = torch.load(abnormal_model_path)
            abnormal_model.load_state_dict(state_dict)
            abnormal_model.cuda()
            abnormal_model.eval()
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in tqdm(loader):
        vol_axial, vol_sagit, vol_coron, label, abnormal = batch
        
        if train:
            if abnormal_model_path and not abnormal:
                continue
            optimizer.zero_grad()

        if loader.dataset.use_gpu:
            vol_axial, vol_sagit, vol_coron = vol_axial.cuda(), vol_sagit.cuda(), vol_coron.cuda()
            label = label.cuda()
        vol_axial, vol_sagit, vol_coron = Variable(vol_axial), Variable(vol_sagit), Variable(vol_coron)
        label = Variable(label)

        if use_amp:
            with torch.cuda.amp.autocast():
                logit = model.forward(vol_axial, vol_sagit, vol_coron)
                loss = loader.dataset.weighted_loss(logit, label)
        else:
            logit = model.forward(vol_axial, vol_sagit, vol_coron)
            loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)

        pred_npy = pred.data.cpu().numpy()[0][0]

        if abnormal_model_path and not train:
            abnormal_logit = abnormal_model.forward(
                    vol_axial,
                    vol_sagit,
                    vol_coron)
            abnormal_pred = torch.sigmoid(abnormal_logit)
            abnormal_pred_npy = abnormal_pred.data.cpu().numpy()[0][0]
            pred_npy = pred_npy * abnormal_pred_npy

        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            if use_amp:
                if scaler is None:
                    raise ValueError("AMP enabled but GradScaler is None")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches
    
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    if abnormal_model_path and not train:
        del abnormal_model

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, use_gpu, num_workers=8):
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu, num_workers=num_workers)

    model = MRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels


def save_confusion_matrix(labels, preds, save_path, threshold=0.5):
    preds_bin = [1 if p >= threshold else 0 for p in preds]
    cm = metrics.confusion_matrix(labels, preds_bin)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_roc_curve(labels, preds, save_path):
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:0.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
