Train MRNet with EfficientNetB0 (triple-plane: axial/sagittal/coronal), full logging, checkpoints, best model, confusion matrix, ROC curve.

## Data format

- Volumes: `data/{train,valid}/{axial,sagittal,coronal}/*.npy`
- Labels: `labels/{train,valid}-{abnormal|acl|meniscus}.csv`

## Run

Run with default config:

```bash
python src_mrnet/train.py
```

Run with config file:

```bash
python src_mrnet/train.py --config src_mrnet/configs/default.json
```

Run with config + CLI override:

```bash
python src_mrnet/train.py --config src_mrnet/configs/default.json --task acl --epochs 30 --run_dir runs/acl_effb0
```

## Outputs

Inside `run_dir`:

- `train.log`: detailed train log
- `resolved_config.json`: config used for this run
- `history.csv`: epoch metrics history
- `checkpoints/last_checkpoint.pth`
- `checkpoints/checkpoint_epoch_XXX.pth`
- `best_model.pth`
- `best_metrics.json`
- `figures/best_confusion_matrix.png`
- `figures/best_roc_curve.png` (if validation has both classes)
- `figures/training_curves.png`
