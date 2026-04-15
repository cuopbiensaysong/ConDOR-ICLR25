#!/usr/bin/env python3
"""Copy or move non-monotone label subjects into a parallel export folder.

Uses the same monotonicity rule as non_monotonic_labels.py. For each affected
subject, transfers train/val/test *.pt files under {split}_data, {split}_label,
{split}_age.

When CT_train.csv / CT-valid.csv / CT_test.csv are present (current export layout),
rows for non-monotone subjects are moved to --out; monotone rows stay in --dir.
Legacy pooled CT.csv is still supported if the split CSVs are absent. Runs before
tensor move/copy so alignment is verified against the full export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from non_monotonic_labels import scan_split
from view_data import load_subject, list_subject_stems

SPLITS = ("train", "val", "test")
KINDS = ("data", "label", "age")
CT_BY_SPLIT = {"train": "CT_train.csv", "val": "CT-valid.csv", "test": "CT_test.csv"}


def folder_name(split: str, kind: str) -> str:
    return f"{split}_{kind}"


def transfer_pt(
    src_root: Path,
    dst_root: Path,
    split: str,
    stem: str,
    *,
    move: bool,
) -> None:
    for kind in KINDS:
        name = folder_name(split, kind)
        src = src_root / name / f"{stem}.pt"
        dst_dir = dst_root / name
        dst = dst_dir / f"{stem}.pt"
        if not src.is_file():
            print(f"warning: missing {src}", file=sys.stderr)
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))


def default_out_dir(src: Path) -> Path:
    return src.parent / f"{src.name}_non_monotonic"


def _all_stems_sorted(roots: tuple[Path, ...]) -> list[str]:
    found: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for split in SPLITS:
            found.update(list_subject_stems(root, split))
    return sorted(found)


def _locate_subject(roots: tuple[Path, ...], stem: str) -> tuple[str, Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for split in SPLITS:
            if (root / f"{split}_data" / f"{stem}.pt").is_file():
                return split, root
    raise FileNotFoundError(f"No {stem}.pt under train/val/test *_data in {roots}")


def _verify_block(
    block: pd.DataFrame,
    x: "object",
    y: "object",
    a: "object",
    node_cols: list[str],
    *,
    stem: str,
) -> None:
    n = int(x.shape[0])
    if len(block) != n:
        raise ValueError(f"{stem}: CT block length {len(block)} != tensor visits {n}")
    ages = a.numpy().reshape(-1).astype(np.float64)
    labs = y.numpy().reshape(-1).astype(np.int64)
    xv = x.numpy().astype(np.float64)
    if not np.allclose(block["age"].to_numpy(dtype=np.float64), ages, rtol=0, atol=1e-4):
        raise ValueError(f"{stem}: CT.csv ages do not match train_age tensor")
    if not (block["label"].to_numpy(dtype=np.int64) == labs).all():
        raise ValueError(f"{stem}: CT.csv labels do not match train_label tensor")
    for j, col in enumerate(node_cols):
        if not np.allclose(block[col].to_numpy(dtype=np.float64), xv[:, j], rtol=1e-5, atol=1e-5):
            raise ValueError(f"{stem}: CT.csv {col} does not match tensor column {j}")


def _load_ct_tables(src_root: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-split CT CSVs if present, else single CT.csv. Returns (dfs_by_split, mode)."""
    train_p = src_root / CT_BY_SPLIT["train"]
    if train_p.is_file():
        hdr = pd.read_csv(train_p, nrows=0)
        dfs = {}
        for sp in SPLITS:
            p = src_root / CT_BY_SPLIT[sp]
            dfs[sp] = pd.read_csv(p) if p.is_file() else pd.DataFrame(columns=hdr.columns)
        return dfs, "split"
    legacy = src_root / "CT.csv"
    if legacy.is_file():
        df = pd.read_csv(legacy)
        return {"pooled": df}, "legacy"
    raise FileNotFoundError("No CT_train.csv or CT.csv under %s" % src_root)


def split_export_ct_csv(
    src_root: Path,
    dst_root: Path,
    bad_stems: set[str],
    *,
    roots_for_tensors: tuple[Path, ...],
) -> tuple[int, int]:
    """Split CT data into monotone (rewrite src) and non-monotone (dst). Returns (n_mono_rows, n_nonmono_rows)."""
    tables, mode = _load_ct_tables(src_root)
    stems = _all_stems_sorted(roots_for_tensors)
    if not stems:
        raise ValueError("No subject tensors found; cannot align CT CSVs")

    mono_by: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    nonmono_by: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    cursors = {s: 0 for s in SPLITS}
    idx_pooled = 0
    pooled_df: pd.DataFrame | None = None

    if mode == "legacy":
        pooled_df = tables["pooled"]
        node_cols = [c for c in pooled_df.columns if c.startswith("Node ")]
        if not node_cols:
            raise ValueError("CT.csv has no 'Node *' region columns")
    else:
        sample = tables["train"]
        if len(sample.columns) == 0:
            sample = pd.read_csv(src_root / CT_BY_SPLIT["train"], nrows=0)
        node_cols = [c for c in sample.columns if c.startswith("Node ")]
        if not node_cols:
            raise ValueError("CT split CSVs have no 'Node *' region columns")

    col_order = ["age", "label"] + node_cols

    for stem in stems:
        split, root = _locate_subject(roots_for_tensors, stem)
        x, y, a = load_subject(root, split, stem)
        n = int(x.shape[0])
        if mode == "split":
            df_sp = tables[split]
            c = cursors[split]
            block = df_sp.iloc[c : c + n].copy()
            cursors[split] = c + n
        else:
            assert pooled_df is not None
            block = pooled_df.iloc[idx_pooled : idx_pooled + n].copy()
            idx_pooled += n
        _verify_block(block, x, y, a, node_cols, stem=stem)
        if stem in bad_stems:
            nonmono_by[split].append(block)
        else:
            mono_by[split].append(block)

    if mode == "split":
        for sp in SPLITS:
            df_sp = tables[sp]
            if cursors[sp] != len(df_sp):
                raise ValueError(
                    f"{CT_BY_SPLIT[sp]} row count {len(df_sp)} does not match tensor visits {cursors[sp]}"
                )
    else:
        assert pooled_df is not None
        if idx_pooled != len(pooled_df):
            raise ValueError(
                f"CT.csv row count {len(pooled_df)} does not match sum of tensor visits {idx_pooled}"
            )

    dst_root.mkdir(parents=True, exist_ok=True)

    n_mono = 0
    n_non = 0
    if mode == "split":
        for sp in SPLITS:
            m_parts = mono_by[sp]
            nm_parts = nonmono_by[sp]
            mdf = pd.concat(m_parts, ignore_index=True) if m_parts else pd.DataFrame(columns=col_order)
            ndf = pd.concat(nm_parts, ignore_index=True) if nm_parts else pd.DataFrame(columns=col_order)
            mdf.to_csv(src_root / CT_BY_SPLIT[sp], index=False)
            ndf.to_csv(dst_root / CT_BY_SPLIT[sp], index=False)
            n_mono += len(mdf)
            n_non += len(ndf)
    else:
        flat_mono = [x for sp in SPLITS for x in mono_by[sp]]
        flat_non = [x for sp in SPLITS for x in nonmono_by[sp]]
        mono_df = pd.concat(flat_mono, ignore_index=True) if flat_mono else pd.DataFrame(columns=col_order)
        nonmono_df = pd.concat(flat_non, ignore_index=True) if flat_non else pd.DataFrame(columns=col_order)
        mono_df.to_csv(src_root / "CT.csv", index=False)
        nonmono_df.to_csv(dst_root / "CT.csv", index=False)
        n_mono = len(mono_df)
        n_non = len(nonmono_df)

    map_path = src_root / "node_region_map.csv"
    if map_path.is_file():
        shutil.copy2(map_path, dst_root / "node_region_map.csv")

    return n_mono, n_non


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=Path("data4condor"),
        help="Source export root (train_data, train_label, ...).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination export root. Default: <dir>_non_monotonic next to --dir.",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files from source instead of copying (removes them from --dir).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Write JSON listing stems per split (default: <out>/non_monotonic_manifest.json).",
    )
    p.add_argument(
        "--no-split-ct",
        action="store_true",
        help="Do not split CT files (default: split when CT_train.csv or CT.csv exists under --dir).",
    )
    args = p.parse_args()

    src_root = args.dir.resolve()
    if not src_root.is_dir():
        print("Source directory not found:", src_root, file=sys.stderr)
        sys.exit(1)

    dst_root = (args.out.resolve() if args.out else default_out_dir(src_root))
    manifest_path = args.manifest.resolve() if args.manifest else (dst_root / "non_monotonic_manifest.json")

    if dst_root == src_root:
        print("--out must differ from --dir.", file=sys.stderr)
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[str]] = {}
    total = 0
    bad_stems: set[str] = set()
    for split in SPLITS:
        bad_rows = scan_split(src_root, split)
        stems = sorted(row["stem"] for row in bad_rows)
        manifest[split] = stems
        bad_stems.update(stems)
        total += len(stems)

    ct_mono_rows: int | None = None
    ct_nonmono_rows: int | None = None
    has_ct = (src_root / "CT_train.csv").is_file() or (src_root / "CT.csv").is_file()
    do_ct = (not args.no_split_ct) and has_ct
    if do_ct:
        # Only src_root: destination may already contain tensors from a prior run.
        ct_mono_rows, ct_nonmono_rows = split_export_ct_csv(
            src_root,
            dst_root,
            bad_stems,
            roots_for_tensors=(src_root,),
        )
        if (src_root / "CT_train.csv").is_file():
            print(f"CT split CSVs: monotone -> {src_root} ({ct_mono_rows} rows total)")
            print(f"CT split CSVs: non-monotone -> {dst_root} ({ct_nonmono_rows} rows total)")
        else:
            print(f"CT.csv: monotone rows -> {src_root / 'CT.csv'} ({ct_mono_rows} rows)")
            print(f"CT.csv: non-monotone rows -> {dst_root / 'CT.csv'} ({ct_nonmono_rows} rows)")

    for split in SPLITS:
        stems = manifest[split]
        for stem in stems:
            transfer_pt(src_root, dst_root, split, stem, move=args.move)
        print(f"{split}: {len(stems)} subject(s) -> {dst_root}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "source": str(src_root),
        "destination": str(dst_root),
        "move": args.move,
        "subjects_by_split": manifest,
        "total_subjects": total,
    }
    if ct_mono_rows is not None:
        payload["ct_csv"] = {
            "monotone_rows": ct_mono_rows,
            "non_monotone_rows": ct_nonmono_rows,
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Done. Total subject-split transfers: {total} ({'move' if args.move else 'copy'}).")


if __name__ == "__main__":
    main()
