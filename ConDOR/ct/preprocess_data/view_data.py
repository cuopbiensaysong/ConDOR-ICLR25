#!/usr/bin/env python3
"""Inspect ConDOR tensors from export_condor_from_csv.py (train/val/test_* folders)."""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

DEFAULT_DATA_DIR = Path(
    "/home/nvidia-lab/ai4life/thaind2/brain/ConDOR-ICLR25/ConDOR/ct/preprocess_data/data4condor"
)

LABEL_NAMES = {0: "CN", 1: "MCI", 2: "Dementia"}


def list_subject_stems(data_dir: Path, split: str) -> list:
    folder = data_dir / f"{split}_data"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.pt"))


def _torch_load_tensor(path: Path):
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=True, **kw)
    except TypeError:
        return torch.load(path, **kw)


def load_subject(data_dir: Path, split: str, stem: str):
    base = data_dir / f"{split}_data" / f"{stem}.pt"
    lab = data_dir / f"{split}_label" / f"{stem}.pt"
    age = data_dir / f"{split}_age" / f"{stem}.pt"
    return _torch_load_tensor(base), _torch_load_tensor(lab), _torch_load_tensor(age)


def print_random_sample(data_dir: Path, split: str, rng: random.Random) -> None:
    stems = list_subject_stems(data_dir, split)
    print(f"\n--- Random sample: {split} ({len(stems)} subjects) ---")
    if not stems:
        print("(no subjects in this split)")
        return
    stem = rng.choice(stems)
    x, y, a = load_subject(data_dir, split, stem)
    print(f"file stem: {stem}.pt")
    print(f"data.shape (visits, regions): {tuple(x.shape)}")
    print(f"ages (per visit): {a.numpy()}")
    labels_np = y.numpy()
    label_strs = [LABEL_NAMES.get(int(i), str(int(i))) for i in labels_np]
    print(f"labels (per visit, int): {labels_np}")
    print(f"labels (per visit, name): {label_strs}")


def visits_per_subject_stats(data_dir: Path, splits: tuple) -> pd.Series:
    counts = []
    for split in splits:
        for stem in list_subject_stems(data_dir, split):
            x, _, _ = load_subject(data_dir, split, stem)
            n_visits = int(x.shape[0])
            counts.append({"split": split, "stem": stem, "num_visits": n_visits})
    if not counts:
        return pd.Series(dtype=float)
    return pd.DataFrame(counts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Export root (contains train_* / val_* / test_* folders).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random subject pick.")
    args = p.parse_args()

    data_dir = args.dir.resolve()
    if not data_dir.is_dir():
        print("Directory not found:", data_dir, file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    for split in ("train", "val", "test"):
        print_random_sample(data_dir, split, rng)

    splits = ("train", "val", "test")
    df = visits_per_subject_stats(data_dir, splits)
    print("\n--- Visits per subject (all exported subjects) ---")
    if df.empty:
        print("No .pt files found under train/val/test.")
        return
    print(df.groupby("split")["num_visits"].agg(["count", "mean", "std", "min", "max"]).to_string())
    s = df["num_visits"]
    print("\n--- Pooled (train + val + test) ---")
    print(f"subjects (rows): {len(s)}")
    print(f"mean : {s.mean():.4f}")
    print(f"std  : {s.std():.4f}")
    print(f"min  : {int(s.min())}")
    print(f"max  : {int(s.max())}")
    print("\n--- describe() pooled ---")
    print(s.describe())


if __name__ == "__main__":
    main()


# Result 

'''
--- Random sample: train (493 subjects) ---
file stem: 130_S_0886.pt
data.shape (visits, regions): (4, 68)
ages (per visit): [71.3 71.8 72.3 73.3]
labels (per visit, int): [0 0 0 0]
labels (per visit, name): ['CN', 'CN', 'CN', 'CN']

--- Random sample: val (70 subjects) ---
file stem: 114_S_0410.pt
data.shape (visits, regions): (3, 68)
ages (per visit): [61.3 61.7 62.2]
labels (per visit, int): [1 1 1]
labels (per visit, name): ['MCI', 'MCI', 'MCI']

--- Random sample: test (141 subjects) ---
file stem: 123_S_1300.pt
data.shape (visits, regions): (6, 68)
ages (per visit): [73.4 73.9 74.4 74.9 75.9 76.9]
labels (per visit, int): [1 1 1 1 1 1]
labels (per visit, name): ['MCI', 'MCI', 'MCI', 'MCI', 'MCI', 'MCI']

--- Visits per subject (all exported subjects) ---
       count      mean       std  min  max
split                                     
test     141  4.000000  1.388730    2    7
train    493  4.093306  1.409685    2    8
val       70  4.300000  1.322602    2    7

--- Pooled (train + val + test) ---
subjects (rows): 704
mean : 4.0952
std  : 1.3973
min  : 2
max  : 8

--- describe() pooled ---
count    704.000000
mean       4.095170
std        1.397327
min        2.000000
25%        3.000000
50%        4.000000
75%        5.000000
max        8.000000
Name: num_visits, dtype: float64
'''