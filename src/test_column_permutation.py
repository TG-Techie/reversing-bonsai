# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Explicit test for input-column permutation between Bonsai-unpacked and Qwen3-base.

Background. The greedy ROW-permutation search in compare_unpacked_vs_qwen3.py
already shows row reorder doesn't help (best_perm_cos == identity_cos). But
it doesn't test column permutation, which is the natural axis for a
"reorder to cluster like-weights into the same 128-block" trick — the trick
is graph-equivalent if the same permutation is applied to the residual
stream's producers and consumers.

Per shared 2D tensor we report:
  * pearson_col_id    — per-column mean|w| Pearson, identity order
  * spearman_col      — Spearman rank correlation of per-column mean|w|
  * sorted_pearson    — Pearson after independently sorting both column
                        sequences by magnitude (≈1 means "same column-
                        magnitude DISTRIBUTION, possibly permuted")
  * ks_sorted         — KS distance between the two sorted sequences (small
                        means same distribution)
  * top10_overlap     — fraction of the top-10% loudest columns that are the
                        same column index in Bonsai and Qwen3 (compared to
                        chance = 10%)

Reading guide:
  - low pearson_col_id, high spearman_col   -> columns permuted, magnitude
    structure preserved under reorder.
  - low pearson_col_id, low  spearman_col   -> column magnitudes really are
    decorrelated; no permutation can recover Qwen3's profile.
  - low pearson_col_id, low  spearman_col,
    BUT sorted_pearson ≈ 1                  -> SAME multiset of column
    magnitudes, in totally different positions: still consistent with
    permutation; identifying it would need joint cross-tensor search.

Usage:
    uv run python src/test_column_permutation.py \
        models/unpacked/model.safetensors \
        models/base/model-00001-of-00002.safetensors \
        --filter "model.layers.0."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from compare_unpacked_vs_qwen3 import load_tensor, list_tensor_names


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / den) if den else float("nan")


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size)
    return ranks


def ks_two_sample_sorted(xs: np.ndarray, ys: np.ndarray) -> float:
    """Two-sample KS for vectors of equal length, both already vectors."""
    rng = np.random.default_rng(0)
    if xs.size > 100_000:
        xs = rng.choice(xs, size=100_000, replace=False)
    if ys.size > 100_000:
        ys = rng.choice(ys, size=100_000, replace=False)
    a = np.sort(xs)
    b = np.sort(ys)
    combined = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, combined, side="right") / a.size
    cdf_b = np.searchsorted(b, combined, side="right") / b.size
    return float(np.abs(cdf_a - cdf_b).max())


def compare_one(name: str, bp: Path, qp: Path) -> dict:
    bw = load_tensor(bp, name)
    qw = load_tensor(qp, name)
    if bw is None or qw is None:
        return {"name": name, "error": "missing"}
    if bw.ndim != 2 or bw.shape != qw.shape:
        return {"name": name, "error": f"shape {bw.shape} vs {qw.shape}"}
    bw = np.abs(bw.astype(np.float32))
    qw = np.abs(qw.astype(np.float32))
    bcol = bw.mean(axis=0)  # one number per input column
    qcol = qw.mean(axis=0)
    brow = bw.mean(axis=1)  # one number per output row
    qrow = qw.mean(axis=1)

    out = {
        "name":             name,
        "shape":            list(bw.shape),
        # Column tests
        "pearson_col_id":   pearson(bcol, qcol),
        "spearman_col":     pearson(rankdata(bcol), rankdata(qcol)),
        "sorted_pearson_col": pearson(np.sort(bcol), np.sort(qcol)),
        "ks_sorted_col":    ks_two_sample_sorted(bcol, qcol),
        # Row tests (sanity check; we already know identity row corr is high)
        "pearson_row_id":   pearson(brow, qrow),
        "spearman_row":     pearson(rankdata(brow), rankdata(qrow)),
    }
    # Top-10% loudest column overlap
    n = bcol.size
    k = max(1, n // 10)
    top_b = set(np.argsort(-bcol)[:k].tolist())
    top_q = set(np.argsort(-qcol)[:k].tolist())
    out["top10_overlap"] = len(top_b & top_q) / k
    out["top10_chance"] = k / n  # would be 0.10 = 10%
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bonsai")
    ap.add_argument("base")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    bp = Path(args.bonsai)
    qp = Path(args.base)
    common = sorted(set(list_tensor_names(bp)) & set(list_tensor_names(qp)))
    rows = []
    for n in common:
        if args.filter and args.filter not in n:
            continue
        r = compare_one(n, bp, qp)
        if "error" in r:
            continue
        if len(r["shape"]) != 2:
            continue
        rows.append(r)
        print(
            f"{n:60s}  shape={r['shape']}\n"
            f"  col: pearson_id={r['pearson_col_id']:+.4f}  "
            f"spearman={r['spearman_col']:+.4f}  "
            f"sorted_pearson={r['sorted_pearson_col']:+.4f}  "
            f"KS_sorted={r['ks_sorted_col']:.4f}  "
            f"top10%_overlap={r['top10_overlap']*100:.2f}% (chance 10%)\n"
            f"  row: pearson_id={r['pearson_row_id']:+.4f}  "
            f"spearman={r['spearman_row']:+.4f}"
        )

    if not rows:
        print("no 2D tensors matched the filter")
        return

    print("\n== AGGREGATE ==")
    keys = [
        ("pearson_col_id",     "per-col Pearson (identity)"),
        ("spearman_col",       "per-col Spearman"),
        ("sorted_pearson_col", "per-col Pearson after sort"),
        ("ks_sorted_col",      "per-col KS(sorted)"),
        ("top10_overlap",      "top-10% loudest col overlap"),
        ("pearson_row_id",     "per-row Pearson (identity)"),
        ("spearman_row",       "per-row Spearman"),
    ]
    for k, label in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"  {label:38s}  mean={vals.mean():+.4f}  min={vals.min():+.4f}  max={vals.max():+.4f}")


if __name__ == "__main__":
    main()
