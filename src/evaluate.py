import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

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
    device = next(model.parameters()).device

    if train:
        model.train()
    else:
        if abnormal_model_path:
            abnormal_model = TripleMRNet(backbone=model.backbone)
            state_dict = torch.load(abnormal_model_path, map_location=device)
            abnormal_model.load_state_dict(state_dict)
            abnormal_model.to(device)
            abnormal_model.eval()
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in tqdm(loader):
        vol_axial, vol_sagit, vol_coron, label, abnormal = batch
        abnormal_flag = bool(abnormal.item()) if torch.is_tensor(abnormal) else bool(abnormal)
        
        if train:
            if abnormal_model_path and not abnormal_flag:
                continue
            optimizer.zero_grad()

        if loader.dataset.use_gpu:
            vol_axial = vol_axial.cuda(non_blocking=True)
            vol_sagit = vol_sagit.cuda(non_blocking=True)
            vol_coron = vol_coron.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        vol_axial, vol_sagit, vol_coron = Variable(vol_axial), Variable(vol_sagit), Variable(vol_coron)
        label = Variable(label)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(vol_axial, vol_sagit, vol_coron)
                    loss = loader.dataset.weighted_loss(logit, label)
            else:
                logit = model(vol_axial, vol_sagit, vol_coron)
                loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)

        pred_npy = pred.data.cpu().numpy()[0][0]

        if abnormal_model_path and not train:
            with torch.no_grad():
                abnormal_logit = abnormal_model(vol_axial, vol_sagit, vol_coron)
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

    if num_batches == 0:
        raise RuntimeError("No batches processed. Check labels/data and abnormal filtering.")
    avg_loss = total_loss / num_batches
    
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    if abnormal_model_path and not train:
        del abnormal_model

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, use_gpu, data_dir="data", labels_dir=None, num_workers=4):
    train_loader, valid_loader = load_data(
        diagnosis, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    model = TripleMRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    else:
        raise ValueError("split must be 'train' or 'valid'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)