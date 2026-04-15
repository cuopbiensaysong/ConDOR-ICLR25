#!/usr/bin/env python3
"""Summarize long-format cortical-thickness CSV for ConDOR CT prep.

Export writes CT_train.csv, CT-valid.csv, and CT_test.csv under the output directory
(see export_condor_from_csv.py). Use --match_export for train-split min/max flags
consistent with that pipeline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import export_condor_from_csv as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default='/home/nvidia-lab/ai4life/thaind2/brain/data/ConDOR_Training_Data_Final.csv')
    p.add_argument("--out", type=Path, default=None, help="Optional text report path")
    p.add_argument(
        "--match_export",
        action="store_true",
        help="Apply export_condor_from_csv cleaning/split; suggest main.py flags from train visits only.",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="With --match_export: validation fraction (default 0.1).",
    )
    p.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="With --match_export: test fraction (default 0.2).",
    )
    p.add_argument("--seed", type=int, default=10, help="With --match_export: RNG seed for stratified split.")
    p.add_argument("--classes", type=int, default=3, help="With --match_export: ordinal classes (default 3).")
    p.add_argument(
        "--dx_map",
        type=str,
        default=None,
        help="With --match_export: DX map, e.g. CN:0,MCI:1,Dementia:2",
    )
    p.add_argument(
        "--keep_non_monotonic",
        action="store_true",
        help="With --match_export: do not drop non-monotone subjects (matches export --keep_non_monotonic).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    regions = ec.region_columns(df)
    n_region = len(regions)

    lines: list[str] = []
    L = lines.append

    L(f"File: {args.csv}")
    L(f"Rows: {len(df)}")
    L(f"Columns: {len(df.columns)}")
    L(f"Region columns (*TA): {n_region}")
    if n_region != 68:
        L(f"  NOTE: expected 68 for this pipeline; got {n_region}.")
    L(f"  First / last: {regions[0]!r}, {regions[-1]!r}")
    L("")

    L("Subjects (PTID): %d unique" % df["PTID"].nunique())
    vc = df.groupby("PTID").size()
    L("Visits per subject: min=%d median=%.1f max=%d" % (vc.min(), vc.median(), vc.max()))
    ge2 = int((vc >= 2).sum())
    L("Subjects with >= 2 visits (usable for TDM): %d / %d" % (ge2, len(vc)))
    L("")

    if "VISCODE" in df.columns:
        L("VISCODE top counts:")
        for k, v in df["VISCODE"].value_counts().head(15).items():
            L(f"  {k}: {v}")
        L("")

    if "DX" in df.columns:
        L("DX value counts:")
        for k, v in df["DX"].value_counts().items():
            L(f"  {k!r}: {v}")
        L("")

    dup_exam = df.duplicated(subset=["PTID", "EXAMDATE"], keep=False).sum()
    dup_vis = df.duplicated(subset=["PTID", "VISCODE"], keep=False).sum()
    L("Duplicate rows (PTID, EXAMDATE): %d row occurrences flagged" % dup_exam)
    L("Duplicate rows (PTID, VISCODE): %d row occurrences flagged" % dup_vis)
    L("")

    na_region = df[regions].isna().sum()
    na_any = int(na_region.sum())
    L("Missing values in region columns (total cells): %d" % na_any)
    if na_any and na_region.max() > 0:
        worst = na_region.nlargest(8)
        L("  Worst columns:")
        for c, n in worst.items():
            if n:
                L(f"    {c}: {n}")
    L("")

    vals = df[regions].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(vals)
    if not finite.all():
        L("Non-finite region values: %d" % (~finite).sum())
    v = vals[finite]
    if v.size:
        L("All regions pooled: min=%.6g max=%.6g" % (v.min(), v.max()))
    L("AGE: min=%.4g max=%.4g" % (df["AGE"].min(), df["AGE"].max()))
    L("")

    if not args.match_export:
        L("WARNING: The following flags use the RAW pooled table (all visits, no cleaning).")
        L("  export_condor_from_csv drops rows with missing regions/AGE/DX, excludes <2-visit subjects,")
        L("  optionally drops non-monotone label trajectories, and suggests --data_min/--age_min from TRAIN tensors only.")
        L("  For export-consistent bounds, re-run with --match_export.")
        L("")
        L("Suggested main.py flags (raw pooled — not export-equivalent):")
        L("  --num_node %d" % n_region)
        if v.size:
            L("  --data_min %.6g  --data_max %.6g" % (float(v.min()), float(v.max())))
        else:
            L("  --data_min (n/a)  --data_max (n/a)")
        L("  --age_min %.4g  --age_max %.4g" % (float(df["AGE"].min()), float(df["AGE"].max())))
    else:
        if not (0.0 < args.test_fraction < 1.0) or not (0.0 <= args.val_fraction < 1.0):
            print("Invalid val/test fractions.", file=sys.stderr)
            sys.exit(1)
        train_fraction = 1.0 - args.val_fraction - args.test_fraction
        if train_fraction <= 0:
            print("val_fraction + test_fraction must be < 1.", file=sys.stderr)
            sys.exit(1)

        if args.dx_map:
            dx_map = ec.parse_dx_map(args.dx_map)
        else:
            dx_map = ec.default_dx_map(args.classes)

        subjects, n_skip = ec.prepare_subjects(df, regions, dx_map)
        ptids_all = sorted(subjects.keys())
        if not args.keep_non_monotonic:
            ptids = sorted(
                p for p in ptids_all if ec.subject_labels_monotone_nondecreasing(subjects[p], dx_map)
            )
            n_drop = len(ptids_all) - len(ptids)
        else:
            ptids = ptids_all
            n_drop = 0

        strata = {p: ec.subject_stratum_group(subjects[p]) for p in ptids}
        unknown = [p for p, s in strata.items() if s == "Other"]
        if unknown:
            print(
                "Cannot stratify (unmapped DX for subject group): %s" % unknown[:5],
                file=sys.stderr,
            )
            sys.exit(1)

        rng = np.random.default_rng(args.seed)
        train_ids, val_ids, test_ids = ec.stratified_train_val_test_split(
            ptids,
            strata,
            rng,
            train_fraction,
            args.val_fraction,
            args.test_fraction,
        )

        tmin = tmax = None
        amin = amax = None
        n_train_visits = 0
        for pid in train_ids:
            g = subjects[pid]
            n_train_visits += len(g)
            arr = g[regions].to_numpy(dtype=np.float64, copy=False)
            ag = g["AGE"].to_numpy(dtype=np.float64, copy=False)
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            tmin = lo if tmin is None else min(tmin, lo)
            tmax = hi if tmax is None else max(tmax, hi)
            loa, hia = float(np.nanmin(ag)), float(np.nanmax(ag))
            amin = loa if amin is None else min(amin, loa)
            amax = hia if amax is None else max(amax, hia)

        L("--match_export summary (same cleaning/split recipe as export_condor_from_csv; train bounds from visits):")
        L("  Subjects after prepare_subjects: %d  skipped: %d" % (len(ptids_all), n_skip))
        if not args.keep_non_monotonic:
            L("  Non-monotone subjects dropped: %d  remaining: %d" % (n_drop, len(ptids)))
        L("  Train / val / test subjects: %d / %d / %d" % (len(train_ids), len(val_ids), len(test_ids)))
        L("  Train visit rows: %d" % n_train_visits)
        L("")
        L("Suggested main.py flags (train visits, export-consistent):")
        L("  --num_node %d" % n_region)
        if tmin is not None and tmax is not None:
            L("  --data_min %.6g  --data_max %.6g" % (tmin, tmax))
        else:
            L("  --data_min (n/a)  --data_max (n/a)")
        if amin is not None and amax is not None:
            L("  --age_min %.4g  --age_max %.4g" % (amin, amax))
        else:
            L("  --age_min (n/a)  --age_max (n/a)")
        L("  --test_num %d" % len(test_ids))
        L("")
        L("After export, ordinal regression uses CT_train.csv only (plus CT-valid.csv / CT_test.csv for reference).")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("Wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
