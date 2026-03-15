import os
import time
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from models.Enhanced_MRNet import MRnet
from config import config
from utils import _train_model, _evaluate_model, _get_lr

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""

def train(config : dict):
    """
    Function where actual training takes place

    Args:
        config (dict) : Configuration to train with
    """
    
    # 1. SETUP THƯ MỤC LƯU WEIGHTS
    save_folder = './weights/{}'.format(config['task'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Directory {save_folder} created!")

    # Định nghĩa đường dẫn file
    checkpoint_path = os.path.join(save_folder, 'last_checkpoint.pth')
    best_model_path = os.path.join(save_folder, 'best_model.pth')

    print('Starting to Train Model...')
    train_loader, val_loader, test_loader = load_data(
        config['task'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        target_slices=config.get('target_slices', 32),
        image_size=config.get('image_size', 224),
    )

    # Lay pos_weight tu dataset
    train_wts = train_loader.dataset.pos_weight
    val_wts = val_loader.dataset.pos_weight

    print('Initializing Model...')
    model = MRnet()
    if torch.cuda.is_available():
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=.3, threshold=1e-4
    )
    
    # Các tham số mặc định
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    best_val_auc = float(0)
    
    # --- ĐOẠN CODE MỚI: KIỂM TRA ĐỂ RESUME (CHẠY TIẾP) ---
    # --- KIEM TRA CHECKPOINT DE RESUME ---
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        try:
            # PyTorch >= 2.6 mac dinh weights_only=True, can tat neu checkpoint la tu ban
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint['best_val_auc']

            print(f"Resuming training from epoch {starting_epoch} with Best AUC: {best_val_auc}")
        except Exception as e:
            print(f"Khong load duoc checkpoint, bo qua va train moi. Ly do: {e}")
            print("Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")
    # -----------------------------------------------------

    patience = config['patience']
    log_train = config['log_train']
    log_val = config['log_val']

    print('Starting Training')

    writer = SummaryWriter(comment='lr={} task={}'.format(config['lr'], config['task']))
    t_start_training = time.time()

    # Vòng lặp chính
    for epoch in range(starting_epoch, num_epochs):

        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        # Train
        train_loss, train_auc = _train_model(
            model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_train)

        # Evaluate
        val_loss, val_auc = _evaluate_model(
            model, val_loader, val_criterion,  epoch, num_epochs, writer, current_lr, log_val)

        # Log tensorboard
        writer.add_scalar('Train/Avg Loss', train_loss, epoch)
        writer.add_scalar('Val/Avg Loss', val_loss, epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("Epoch [{}/{}] | train loss : {:.4f} | train auc {:.4f} | val loss {:.4f} | val auc {:.4f} | time {:.2f} s".format(
            epoch, num_epochs, train_loss, train_auc, val_loss, val_auc, delta))

        print('-' * 30)
        writer.flush()

        # --- LƯU MODEL TỐT NHẤT (BEST) ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc,
            }, best_model_path)
        
        # --- LƯU CHECKPOINT ĐỂ RESUME SAU NÀY (LAST) ---
        # Lưu đè lên file này mỗi cuối epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_auc': best_val_auc, # Lưu lại kỷ lục hiện tại
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        # -----------------------------------------------

    t_end_training = time.time()
    print(f'Training finished. Total time: {t_end_training - t_start_training:.2f} s')
    writer.flush()
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
