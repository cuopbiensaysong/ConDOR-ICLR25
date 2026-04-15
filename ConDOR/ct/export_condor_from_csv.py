#!/usr/bin/env python3
"""
Build ConDOR CT folder layout and per-split CT CSVs from long-format cortical thickness CSV.

Writes train_/val_/test_ folders (*_data, *_label, *_age), CT_train.csv, CT-valid.csv,
CT_test.csv, and node_region_map.csv under --out_dir. Subjects are split 70:10:20
(default) into train/val/test, stratified by subject-level disease (same worst-stage
rule as data/statistic_by_disease.py: Dementia > MCI > CN).

CT_train.csv holds training visits only (for ordinal regression in or_batch.py);
val/test visits are in CT-valid.csv and CT_test.csv respectively.

Rows with NaN in any region or missing AGE/DX are dropped per-subject before
counting visits; subjects with fewer than 2 visits are skipped.

By default, subjects whose visit labels are not non-decreasing in chronological
age order are excluded from the main export; use --keep_non_monotonic to disable.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

CT_CSV_BY_SPLIT = {"train": "CT_train.csv", "val": "CT-valid.csv", "test": "CT_test.csv"}
NONMONO_RNG_OFFSET = 7919


def region_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    for req in ("PTID", "AGE", "DX"):
        if req not in cols:
            raise ValueError("Expected column %r in CSV." % req)
    regions = [c for c in cols if c.endswith("TA")]
    if not regions:
        raise ValueError("No cortical thickness columns (*TA) found.")
    return regions


def safe_ptid(ptid: str) -> str:
    s = str(ptid).strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s or "subject"


def default_dx_map(classes: int) -> Dict[str, int]:
    if classes == 3:
        return {"CN": 0, "MCI": 1, "Dementia": 2}
    raise ValueError(
        "Use --classes 3 with default mapping, or pass --dx_map for other layouts."
    )


def parse_dx_map(spec: str) -> Dict[str, int]:
    """Format: CN:0,MCI:1,AD:2"""
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("Each --dx_map entry must be NAME:int, comma-separated.")
        k, v = part.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


# Visit-level DX -> broad group (aligns with data/statistic_by_disease.py / disease_count.py)
SUBJECT_DX_TO_BROAD: Dict[str, str] = {
    "CN": "CN",
    "SMC": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "AD": "Dementia",
    "MCI": "MCI",
    "Dementia": "Dementia",
}


def subject_stratum_group(g: pd.DataFrame) -> str:
    """Worst stage across visits: Dementia > MCI > CN."""
    broad = set()
    for dx in g["DX"]:
        key = str(dx).strip()
        b = SUBJECT_DX_TO_BROAD.get(key, key)
        broad.add(b)
    if "Dementia" in broad:
        return "Dementia"
    if "MCI" in broad:
        return "MCI"
    if "CN" in broad:
        return "CN"
    return "Other"


def _split_sizes(n: int, f_train: float, f_val: float, f_test: float) -> Tuple[int, int, int]:
    """Integer train/val/test counts for n subjects, approximating given fractions."""
    n_test = int(round(f_test * n))
    n_val = int(round(f_val * n))
    n_train = n - n_val - n_test
    if n_train < 0:
        n_val = max(0, n_val + n_train)
        n_train = n - n_val - n_test
    if n_train < 0:
        n_test = max(0, n_test + n_train)
        n_train = n - n_val - n_test
    assert n_train + n_val + n_test == n
    return n_train, n_val, n_test


def stratified_train_val_test_split(
    ptids: List[str],
    strata: Dict[str, str],
    rng: np.random.Generator,
    f_train: float,
    f_val: float,
    f_test: float,
) -> Tuple[set, set, set]:
    """Stratified 3-way split on stratum label (e.g. CN / MCI / Dementia)."""
    train_ids: set = set()
    val_ids: set = set()
    test_ids: set = set()
    by_stratum: Dict[str, List[str]] = {}
    for p in ptids:
        s = strata[p]
        by_stratum.setdefault(s, []).append(p)
    for s, group in by_stratum.items():
        order = list(group)
        rng.shuffle(order)
        n = len(order)
        n_tr, n_va, n_te = _split_sizes(n, f_train, f_val, f_test)
        i0, i1, i2 = 0, n_tr, n_tr + n_va
        train_ids.update(order[i0:i1])
        val_ids.update(order[i1:i2])
        test_ids.update(order[i2:])
    return train_ids, val_ids, test_ids


def prepare_subjects(
    df: pd.DataFrame, regions: List[str], dx_map: Dict[str, int]
) -> Tuple[Dict[str, pd.DataFrame], int]:
    """Return ptid -> sorted clean dataframe (>=2 rows), and count skipped."""
    df = df.copy()
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    skipped = 0
    out: Dict[str, pd.DataFrame] = {}

    for ptid, g in df.groupby("PTID", sort=False):
        g = g.sort_values("EXAMDATE")
        mask = g["EXAMDATE"].notna() & g["AGE"].notna() & g["DX"].notna()
        for c in regions:
            mask &= g[c].notna()
        g = g.loc[mask]
        if g["DX"].apply(lambda x: str(x).strip() not in dx_map).any():
            skipped += 1
            continue
        if len(g) < 2:
            skipped += 1
            continue
        out[str(ptid)] = g

    return out, skipped


def subject_labels_monotone_nondecreasing(g: pd.DataFrame, dx_map: Dict[str, int]) -> bool:
    """True if integer labels are non-decreasing when visits are ordered by age."""
    labels = np.array([dx_map[str(dx).strip()] for dx in g["DX"]], dtype=np.int64)
    ages = g["AGE"].to_numpy(dtype=np.float64)
    order = np.argsort(ages, kind="stable")
    labs = labels[order]
    return bool((labs[1:] >= labs[:-1]).all())


def _ensure_subdirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in (
        "train_data",
        "train_label",
        "train_age",
        "val_data",
        "val_label",
        "val_age",
        "test_data",
        "test_label",
        "test_age",
    ):
        (out_dir / s).mkdir(parents=True, exist_ok=True)


def _write_node_map(out_dir: Path, regions: List[str], node_names: List[str]) -> Path:
    node_map_df = pd.DataFrame(
        {
            "node_index": list(range(1, len(regions) + 1)),
            "node_name": node_names,
            "csv_column": regions,
        }
    )
    path = out_dir / "node_region_map.csv"
    node_map_df.to_csv(path, index=False)
    return path


def _write_ct_csvs(
    out_dir: Path,
    ct_rows: Dict[str, List[dict]],
    col_order: List[str],
    ct_pooled_rows: Optional[List[dict]],
) -> Tuple[int, int, int]:
    """Write CT_train.csv, CT-valid.csv, CT_test.csv; optional pooled CT.csv (same PTID sort as export loop)."""
    counts = []
    for split in ("train", "val", "test"):
        rows = ct_rows[split]
        fname = CT_CSV_BY_SPLIT[split]
        if rows:
            df = pd.DataFrame(rows)
            df = df[col_order]
        else:
            df = pd.DataFrame(columns=col_order)
        df.to_csv(out_dir / fname, index=False)
        counts.append(len(df))
    if ct_pooled_rows is not None:
        if ct_pooled_rows:
            pdf = pd.DataFrame(ct_pooled_rows)
            pdf = pdf[col_order]
        else:
            pdf = pd.DataFrame(columns=col_order)
        pdf.to_csv(out_dir / "CT.csv", index=False)
    return counts[0], counts[1], counts[2]


def _export_cohort(
    out_dir: Path,
    subjects: Dict[str, pd.DataFrame],
    ptids: List[str],
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
    regions: List[str],
    dx_map: Dict[str, int],
    node_names: List[str],
    col_order: List[str],
    write_pooled_ct_csv: bool,
    write_node_map: bool,
    node_map_source: Optional[Path] = None,
) -> Tuple[int, int, int]:
    """
    Write tensors and per-split CT CSVs for the given subject list and split assignment.
    Returns (n_train_rows, n_val_rows, n_test_rows).
    """
    _ensure_subdirs(out_dir)
    if write_node_map:
        _write_node_map(out_dir, regions, node_names)
    elif node_map_source is not None and node_map_source.is_file():
        shutil.copy2(node_map_source, out_dir / "node_region_map.csv")

    ct_rows: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    ct_pooled: Optional[List[dict]] = [] if write_pooled_ct_csv else None

    def export_one(ptid: str, g: pd.DataFrame, split: str) -> None:
        base = safe_ptid(ptid)
        x = g[regions].to_numpy(dtype=np.float32, copy=True)
        labels = np.array([dx_map[str(dx).strip()] for dx in g["DX"]], dtype=np.int64)
        ages = g["AGE"].to_numpy(dtype=np.float32, copy=True)

        stem = base + ".pt"
        torch.save(torch.from_numpy(x), out_dir / (split + "_data") / stem)
        torch.save(torch.from_numpy(labels), out_dir / (split + "_label") / stem)
        torch.save(torch.from_numpy(ages), out_dir / (split + "_age") / stem)

        for i in range(len(g)):
            row = {"age": float(ages[i]), "label": int(labels[i])}
            for j, name in enumerate(node_names):
                row[name] = float(x[i, j])
            ct_rows[split].append(row)
            if ct_pooled is not None:
                ct_pooled.append(row)

    for ptid in ptids:
        g = subjects[ptid]
        if ptid in test_ids:
            split = "test"
        elif ptid in val_ids:
            split = "val"
        else:
            split = "train"
        export_one(ptid, g, split)

    return _write_ct_csvs(out_dir, ct_rows, col_order, ct_pooled)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default='/home/nvidia-lab/ai4life/thaind2/brain/data/ConDOR_Training_Data_Final.csv')
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Validation fraction (default 0.1 for 70/10/20 with --test_fraction 0.2).",
    )
    p.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Test fraction (default 0.2). Train fraction is 1 - val - test.",
    )
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--classes", type=int, default=3, help="Number of ordinal classes (labels are 0..classes-1).")
    p.add_argument(
        "--dx_map",
        type=str,
        default=None,
        help="Override DX mapping, e.g. CN:0,MCI:1,Dementia:2",
    )
    p.add_argument(
        "--write_pooled_ct_csv",
        action="store_true",
        help="Also write legacy CT.csv with all visits pooled (train+val+test order by sorted PTID).",
    )
    p.add_argument(
        "--keep_non_monotonic",
        action="store_true",
        help="Do not exclude subjects with decreasing label over time (default: exclude them from main export).",
    )
    p.add_argument(
        "--non_monotonic_out_dir",
        type=Path,
        default=None,
        help="If set, export excluded non-monotone subjects here with the same layout and separate RNG.",
    )
    args = p.parse_args()

    if not (0.0 < args.test_fraction < 1.0):
        p.error("test_fraction must be in (0, 1)")
    if not (0.0 <= args.val_fraction < 1.0):
        p.error("val_fraction must be in [0, 1)")
    train_fraction = 1.0 - args.val_fraction - args.test_fraction
    if train_fraction <= 0:
        p.error("val_fraction + test_fraction must be < 1 (need positive train fraction)")

    df = pd.read_csv(args.csv)
    regions = region_columns(df)
    n_node = len(regions)
    if n_node != 68:
        print(
            "Note: ConDOR CT examples use 68 DK regions; this CSV has %d region columns."
            % n_node,
            file=sys.stderr,
        )
        raise ValueError("Expected 68 region columns, got %d" % n_node)

    if args.dx_map:
        dx_map = parse_dx_map(args.dx_map)
    else:
        dx_map = default_dx_map(args.classes)

    max_label = max(dx_map.values())
    if max_label >= args.classes:
        p.error("dx_map labels must be in 0..classes-1")
    min_label = min(dx_map.values())
    if min_label != 0:
        p.error("dx_map labels must include 0 as the smallest class index (ordinal 0..K-1).")

    subjects, n_skip = prepare_subjects(df, regions, dx_map)
    ptids_all = sorted(subjects.keys())

    if not args.keep_non_monotonic:
        ptids_mono = [p for p in ptids_all if subject_labels_monotone_nondecreasing(subjects[p], dx_map)]
        ptids_nonmono = [p for p in ptids_all if p not in set(ptids_mono)]
        print(
            "Non-monotone subjects excluded from main export: %d (kept %d)"
            % (len(ptids_nonmono), len(ptids_mono)),
            file=sys.stderr,
        )
        if ptids_nonmono and len(ptids_nonmono) <= 30:
            print("  PTIDs: %s" % ", ".join(ptids_nonmono), file=sys.stderr)
        ptids = sorted(ptids_mono)
    else:
        ptids_nonmono = []
        ptids = ptids_all

    strata = {p: subject_stratum_group(subjects[p]) for p in ptids}
    unknown = [p for p, s in strata.items() if s == "Other"]
    if unknown:
        p.error(
            "Cannot stratify %d subject(s) with unmapped visit DX for subject-level group: %s"
            % (len(unknown), unknown[:5])
        )

    rng = np.random.default_rng(args.seed)
    train_ids, val_ids, test_ids = stratified_train_val_test_split(
        ptids,
        strata,
        rng,
        train_fraction,
        args.val_fraction,
        args.test_fraction,
    )

    if not train_ids:
        p.error("Train split empty; adjust fractions or check data.")
    if not test_ids:
        p.error("Test split empty; lower --val_fraction or check data.")

    node_names = ["Node %d" % (i + 1) for i in range(n_node)]
    col_order = ["age", "label"] + node_names

    n_tr, n_va, n_te = _export_cohort(
        args.out_dir,
        subjects,
        ptids,
        train_ids,
        val_ids,
        test_ids,
        regions,
        dx_map,
        node_names,
        col_order,
        args.write_pooled_ct_csv,
        write_node_map=True,
    )

    ct_parts = []
    for split in ("train", "val", "test"):
        path = args.out_dir / CT_CSV_BY_SPLIT[split]
        if path.is_file():
            ct_parts.append(pd.read_csv(path))
    ct_df = pd.concat(ct_parts, ignore_index=True) if ct_parts else pd.DataFrame(columns=col_order)

    labels_used = set(ct_df["label"].unique().tolist()) if len(ct_df) else set()
    expected = set(range(args.classes))
    if labels_used and not labels_used.issubset(expected):
        p.error("Exported labels %s are not within 0..%d" % (sorted(labels_used), args.classes - 1))
    if labels_used != expected and labels_used:
        print(
            "Warning: not all class indices appear in CT exports (present %s). OrderedModel still runs if 0..K-1 is valid."
            % sorted(labels_used),
            file=sys.stderr,
        )

    node_map_path = args.out_dir / "node_region_map.csv"

    # Suggested ranges from training tensors only
    tmin, tmax = float("inf"), float("-inf")
    amin, amax = float("inf"), float("-inf")
    for ptid in train_ids:
        stem = safe_ptid(ptid) + ".pt"
        x = torch.load(args.out_dir / "train_data" / stem, map_location="cpu")
        a = torch.load(args.out_dir / "train_age" / stem, map_location="cpu")
        tmin = min(tmin, float(x.min()))
        tmax = max(tmax, float(x.max()))
        amin = min(amin, float(a.min()))
        amax = max(amax, float(a.max()))

    print(
        "Subjects exported: %d (train %d, val %d, test %d)"
        % (len(ptids), len(train_ids), len(val_ids), len(test_ids))
    )
    print("Subjects skipped (unknown DX or <2 clean visits): %d" % n_skip)
    print("Stratification (subject-level): %s" % dict(pd.Series(list(strata.values())).value_counts()))
    print(
        "Regions: %d  CT rows — train %d, val %d, test %d (total %d)"
        % (n_node, n_tr, n_va, n_te, n_tr + n_va + n_te)
    )
    print("Wrote:", args.out_dir)
    print("Per-split CT:", ", ".join(CT_CSV_BY_SPLIT[s] for s in ("train", "val", "test")))
    if args.write_pooled_ct_csv:
        print("Also wrote pooled CT.csv")
    print("Node <-> CSV column map:", node_map_path.resolve())
    print("")
    print("Suggested main.py flags:")
    print("  --dir %s" % args.out_dir.resolve())
    print("  --num_node %d" % n_node)
    print("  --classes %d" % args.classes)
    print("  --data_min %.6g  --data_max %.6g" % (tmin, tmax))
    print("  --age_min %.4g  --age_max %.4g" % (amin, amax))
    print("  --test_num %d" % len(test_ids))
    print("(Val split: %d subjects under val_*; ConDOR main.py uses train/test only.)" % len(val_ids))

    if args.non_monotonic_out_dir is not None:
        nm_dir = args.non_monotonic_out_dir.resolve()
        if args.keep_non_monotonic:
            print("Note: --non_monotonic_out_dir set but no exclusions (--keep_non_monotonic).", file=sys.stderr)
        elif not ptids_nonmono:
            print("No non-monotone subjects to export to %s" % nm_dir)
            nm_dir.mkdir(parents=True, exist_ok=True)
            _ensure_subdirs(nm_dir)
            _write_ct_csvs(
                nm_dir,
                {"train": [], "val": [], "test": []},
                col_order,
                [] if args.write_pooled_ct_csv else None,
            )
            if (args.out_dir / "node_region_map.csv").is_file():
                shutil.copy2(args.out_dir / "node_region_map.csv", nm_dir / "node_region_map.csv")
        else:
            strata_nm = {p: subject_stratum_group(subjects[p]) for p in ptids_nonmono}
            unk_nm = [p for p, s in strata_nm.items() if s == "Other"]
            if unk_nm:
                p.error("Non-monotone cohort has unstratifiable subjects: %s" % unk_nm[:5])
            rng_nm = np.random.default_rng(args.seed + NONMONO_RNG_OFFSET)
            tr_nm, va_nm, te_nm = stratified_train_val_test_split(
                sorted(ptids_nonmono),
                strata_nm,
                rng_nm,
                train_fraction,
                args.val_fraction,
                args.test_fraction,
            )
            _export_cohort(
                nm_dir,
                subjects,
                sorted(ptids_nonmono),
                tr_nm,
                va_nm,
                te_nm,
                regions,
                dx_map,
                node_names,
                col_order,
                args.write_pooled_ct_csv,
                write_node_map=False,
                node_map_source=args.out_dir / "node_region_map.csv",
            )
            print("Non-monotone cohort exported to:", nm_dir)


if __name__ == "__main__":
    main()
