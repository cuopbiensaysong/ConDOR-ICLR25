#!/usr/bin/env python3
"""Find subjects whose visit labels decrease over time (impossible if severity is monotone).

Disease stages are ordered CN (0) < MCI (1) < Dementia (2). Visits are ordered by age
before checking consecutive pairs, matching temporal monotonicity of severity.
"""

import argparse
import sys
from pathlib import Path

import torch

from view_data import DEFAULT_DATA_DIR, LABEL_NAMES, list_subject_stems, load_subject


def _argsort_stable(ages: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(ages, stable=True)
    except TypeError:
        return torch.argsort(ages)


def label_sequence_str(labels: torch.Tensor) -> str:
    parts = [LABEL_NAMES.get(int(i), str(int(i))) for i in labels.tolist()]
    return "[" + ", ".join(parts) + "]"


def monotonic_violations(
    ages: torch.Tensor, labels: torch.Tensor
) -> list[tuple[int, int, float, float, int, int]]:
    """Return list of (k, k+1, age_k, age_k1, lab_k, lab_k1) where lab decreases after sort by age."""
    ages = ages.flatten().float()
    labels = labels.flatten().long()
    order = _argsort_stable(ages)
    ages_s = ages[order]
    labs_s = labels[order]
    out: list[tuple[int, int, float, float, int, int]] = []
    for k in range(int(labs_s.numel()) - 1):
        a0, a1 = float(ages_s[k].item()), float(ages_s[k + 1].item())
        l0, l1 = int(labs_s[k].item()), int(labs_s[k + 1].item())
        if l1 < l0:
            out.append((k, k + 1, a0, a1, l0, l1))
    return out


def scan_split(data_dir: Path, split: str) -> list[dict]:
    rows = []
    for stem in list_subject_stems(data_dir, split):
        _, y, a = load_subject(data_dir, split, stem)
        if y.numel() < 2:
            continue
        viol = monotonic_violations(a, y)
        if viol:
            ages = a.flatten().float()
            labels = y.flatten().long()
            order = _argsort_stable(ages)
            rows.append(
                {
                    "stem": stem,
                    "labels_chronological": labels[order].clone(),
                    "ages_chronological": ages[order].clone(),
                    "violations": viol,
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Export root (contains train_* / val_* / test_* folders).",
    )
    args = p.parse_args()

    data_dir = args.dir.resolve()
    if not data_dir.is_dir():
        print("Directory not found:", data_dir, file=sys.stderr)
        sys.exit(1)

    print("Monotonicity: labels must not decrease when visits are sorted by age.")
    print("(0=CN, 1=MCI, 2=Dementia)\n")

    grand_total = 0
    for split in ("train", "val", "test"):
        stems = list_subject_stems(data_dir, split)
        n_subjects = len(stems)
        bad = scan_split(data_dir, split)
        grand_total += len(bad)

        print(f"=== {split} ===")
        print(f"subjects in split: {n_subjects}")
        print(f"subjects with at least one decreasing label step: {len(bad)}")
        if n_subjects:
            print(f"fraction: {len(bad) / n_subjects:.4f}")

        for row in bad:
            stem = row["stem"]
            labs = row["labels_chronological"]
            ages = row["ages_chronological"]
            print(f"\n  {stem}")
            print(f"    ages (chronological):  {ages.numpy()}")
            print(f"    labels (int):          {labs.numpy()}")
            print(f"    labels (name):         {label_sequence_str(labs)}")
            for _, _, age_a, age_b, la, lb in row["violations"]:
                na = LABEL_NAMES.get(la, str(la))
                nb = LABEL_NAMES.get(lb, str(lb))
                print(f"    violation: {na} ({la}) at age {age_a:.2f} -> {nb} ({lb}) at age {age_b:.2f}")

        print()

    print(f"--- Total across train+val+test: {grand_total} subjects with non-monotone labels ---")


if __name__ == "__main__":
    main()