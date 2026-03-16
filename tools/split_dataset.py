import os
import argparse
import random
import shutil
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def move_file(src: str, dst: str, dry_run: bool):
    if dry_run:
        return
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def split_ids(ids, seed, ratios):
    r_train, r_val, r_test = ratios
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--label-root', default='./labels')
    ap.add_argument('--task', default='abnormal')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    r_train, r_val, r_test = args.ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError('ratios must sum to 1.0')

    train_csv = os.path.join(args.label_root, f'train-{args.task}.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(train_csv)

    df = pd.read_csv(train_csv, header=None, names=['id', 'label'])
    df['id'] = df['id'].map(lambda i: str(i).zfill(4))

    train_ids, val_ids, test_ids = split_ids(df['id'].tolist(), args.seed, args.ratios)

    def write_csv(split_name, ids):
        out_csv = os.path.join(args.label_root, f'{split_name}-{args.task}.csv')
        split_df = df[df['id'].isin(ids)][['id', 'label']]
        split_df['id'] = split_df['id'].astype(str).str.lstrip('0').replace('', '0')
        if not args.dry_run:
            split_df.to_csv(out_csv, header=False, index=False)
        return out_csv, len(split_df)

    for split_name, ids in [('train', train_ids), ('valid', val_ids), ('test', test_ids)]:
        for plane in ['axial', 'coronal', 'sagittal']:
            src_dir = os.path.join(args.data_root, 'train', plane)
            dst_dir = os.path.join(args.data_root, split_name, plane)
            ensure_dir(dst_dir)
            for id_ in ids:
                src = os.path.join(src_dir, f'{id_}.npy')
                dst = os.path.join(dst_dir, f'{id_}.npy')
                if not os.path.exists(src):
                    continue
                move_file(src, dst, args.dry_run)

    train_out, n_train = write_csv('train', train_ids)
    val_out, n_val = write_csv('valid', val_ids)
    test_out, n_test = write_csv('test', test_ids)

    print('Done.')
    print('Train:', n_train, '->', train_out)
    print('Valid:', n_val, '->', val_out)
    print('Test :', n_test, '->', test_out)
    if args.dry_run:
        print('Dry run: no files moved or written.')


if __name__ == '__main__':
    main()
