import argparse
import json
import os
import random
import shutil
from datetime import datetime

import pandas as pd


TASKS = ["abnormal", "acl", "meniscus"]
PLANES = ["axial", "coronal", "sagittal"]
SPLITS = ["train", "valid", "test"]


def read_labels(labels_dir):
    labels = {t: {} for t in TASKS}
    for split in SPLITS:
        for task in TASKS:
            path = os.path.join(labels_dir, f"{split}-{task}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing label file: {path}")
            df = pd.read_csv(path, header=None, names=["id", "label"])
            for rid, label in df.values:
                rid = int(rid)
                label = int(label)
                existing = labels[task].get(rid)
                if existing is not None and existing != label:
                    raise ValueError(f"Conflicting labels for id={rid} task={task}")
                labels[task][rid] = label

    common_ids = set.intersection(*(set(labels[t].keys()) for t in TASKS))
    return labels, sorted(common_ids)


def try_stratified_split(ids, y, seed):
    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        return None

    try:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            ids, y, test_size=0.30, random_state=seed, stratify=y
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.50, random_state=seed, stratify=temp_y
        )
        return {"train": train_ids, "valid": val_ids, "test": test_ids}
    except Exception:
        return None


def manual_stratified_split(ids, y, seed):
    rnd = random.Random(seed)
    groups = {0: [], 1: []}
    for rid, label in zip(ids, y):
        groups[int(label)].append(rid)

    for k in groups:
        rnd.shuffle(groups[k])

    split_ids = {s: [] for s in SPLITS}
    ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}

    for label, group in groups.items():
        n = len(group)
        exact = {s: n * ratios[s] for s in SPLITS}
        base = {s: int(exact[s]) for s in SPLITS}
        remain = n - sum(base.values())
        frac_order = sorted(SPLITS, key=lambda s: exact[s] - base[s], reverse=True)
        for s in frac_order:
            if remain <= 0:
                break
            base[s] += 1
            remain -= 1

        idx = 0
        for s in SPLITS:
            split_ids[s].extend(group[idx : idx + base[s]])
            idx += base[s]

    return split_ids


def find_source_split(data_dir, rid_str):
    for split in SPLITS:
        ok = True
        for plane in PLANES:
            path = os.path.join(data_dir, split, plane, f"{rid_str}.npy")
            if not os.path.exists(path):
                ok = False
                break
        if ok:
            return split
    return None


def write_labels(split_ids, labels, out_labels):
    os.makedirs(out_labels, exist_ok=True)
    for split in SPLITS:
        for task in TASKS:
            path = os.path.join(out_labels, f"{split}-{task}.csv")
            with open(path, "w", newline="") as f:
                for rid in split_ids[split]:
                    f.write(f"{rid},{labels[task][rid]}\n")


def copy_or_move_data(split_ids, data_dir, out_data, mode):
    os.makedirs(out_data, exist_ok=True)
    for split in SPLITS:
        for plane in PLANES:
            os.makedirs(os.path.join(out_data, split, plane), exist_ok=True)

    moved = 0
    missing = []
    for split in SPLITS:
        for rid in split_ids[split]:
            rid_str = str(rid).zfill(4)
            src_split = find_source_split(data_dir, rid_str)
            if src_split is None:
                missing.append(rid)
                continue
            for plane in PLANES:
                src = os.path.join(data_dir, src_split, plane, f"{rid_str}.npy")
                dst = os.path.join(out_data, split, plane, f"{rid_str}.npy")
                if mode == "move":
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
            moved += 1

    return moved, missing


def summarize(split_ids, labels, task_for_split):
    summary = {"task_for_split": task_for_split}
    for split in SPLITS:
        ids = split_ids[split]
        summary[split] = {"count": len(ids)}
        for task in TASKS:
            pos = sum(labels[task][rid] for rid in ids)
            neg = len(ids) - pos
            rate = (pos / len(ids)) if ids else 0.0
            summary[split][task] = {"pos": pos, "neg": neg, "pos_rate": rate}
    return summary


def backup_dir(path):
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{path}_backup_{ts}"
    shutil.move(path, backup)
    return backup


def main():
    parser = argparse.ArgumentParser(
        description="Resplit data 70/15/15 optimized for one chosen task."
    )
    parser.add_argument("--task", required=False, choices=TASKS)
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-labels", default="labels_resplit")
    parser.add_argument("--out-data", default="data_resplit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace existing labels/ and data/ (creates backups).",
    )
    args = parser.parse_args()

    if not args.task:
        print("Chon task can chia:")
        for i, t in enumerate(TASKS, 1):
            print(f"{i}. {t}")
        while True:
            choice = input("Nhap so (1-3): ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(TASKS):
                    args.task = TASKS[idx - 1]
                    break
            print("Lua chon khong hop le. Thu lai.")

    labels, ids = read_labels(args.labels_dir)
    y = [labels[args.task][rid] for rid in ids]

    split_ids = try_stratified_split(ids, y, args.seed)
    if split_ids is None:
        split_ids = manual_stratified_split(ids, y, args.seed)

    if args.inplace:
        backup_labels = backup_dir(args.labels_dir)
        backup_data = backup_dir(args.data_dir)
        print(f"Backed up labels to: {backup_labels}")
        print(f"Backed up data to: {backup_data}")
        out_labels = args.labels_dir
        out_data = args.data_dir
    else:
        out_labels = args.out_labels
        out_data = args.out_data

    write_labels(split_ids, labels, out_labels)
    moved, missing = copy_or_move_data(split_ids, args.data_dir, out_data, args.mode)

    summary = summarize(split_ids, labels, args.task)
    summary_path = os.path.join(out_labels, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Task for split: {args.task}")
    print(f"Wrote labels to: {out_labels}")
    print(f"Wrote data to: {out_data} (mode={args.mode}, files={moved})")
    if missing:
        print(f"Missing data for {len(missing)} ids. See split_summary.json for counts.")


if __name__ == "__main__":
    main()
