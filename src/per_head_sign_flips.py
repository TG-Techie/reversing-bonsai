# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Per-head sign-flip analysis for attention projections.

Why: attention projections (q/k/v/o) have a natural per-head row
structure. If the recipe pushes some heads harder than others, we'd
see specific heads with higher flip rates than the per-tensor mean.
That would suggest head-targeted training (e.g. distillation losses
that weight certain heads).

For each q/k/v projection at layer L:
  shape = (head_count * head_dim, hidden) for q/k
  shape = (head_count * head_dim, hidden) for v   (uses head_count_kv)

We slice rows into chunks of head_dim and compute per-head flip rate
relative to Qwen3-base.

For o_proj:
  shape = (hidden, head_count * head_dim) — the per-head structure is
  in the COLUMNS, not the rows. We slice columns instead.

Usage:
    uv run python src/per_head_sign_flips.py models/bonsai/<size>/unpacked/ \
        models/bonsai/<size>/base/ \
        --head-dim 128 --filter "model.layers.0."
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_unpacked_vs_qwen3 import load_tensor, list_tensor_names


def per_head_flip_rates(W_b: np.ndarray, W_q: np.ndarray, head_dim: int, axis: int) -> np.ndarray:
    """Slice W along `axis` into chunks of head_dim and report flip-rate per chunk."""
    n = W_b.shape[axis]
    assert n % head_dim == 0, f"axis size {n} not divisible by head_dim {head_dim}"
    nhead = n // head_dim
    rates = np.zeros(nhead)
    for h in range(nhead):
        if axis == 0:
            a = W_b[h * head_dim:(h + 1) * head_dim].flatten()
            b = W_q[h * head_dim:(h + 1) * head_dim].flatten()
        else:
            a = W_b[:, h * head_dim:(h + 1) * head_dim].flatten()
            b = W_q[:, h * head_dim:(h + 1) * head_dim].flatten()
        nz = (np.abs(a) > 0) & (np.abs(b) > 0)
        if nz.sum() == 0:
            rates[h] = np.nan
        else:
            rates[h] = (np.sign(a[nz]) != np.sign(b[nz])).mean()
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unpacked")
    ap.add_argument("base")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    unp = Path(args.unpacked)
    base = Path(args.base)
    names = sorted(set(list_tensor_names(unp)) & set(list_tensor_names(base)))

    print(f"head_dim = {args.head_dim}")
    print(f"{'tensor':70s}  n_heads  mean   std    min    max   range")
    by_proj = defaultdict(list)
    by_head_global = defaultdict(list)  # for q_proj only — head index -> [rates]
    for n in names:
        if args.filter and args.filter not in n:
            continue
        if not n.endswith(".weight"):
            continue
        if "norm" in n or "embed" in n:
            continue
        if not any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        u = load_tensor(unp, n)
        b = load_tensor(base, n)
        if u is None or b is None or u.ndim != 2:
            continue
        # axis: rows for q/k/v (out dim is head_count*head_dim); cols for o (in dim)
        axis = 1 if "o_proj" in n else 0
        try:
            rates = per_head_flip_rates(b, u, args.head_dim, axis)
        except AssertionError as e:
            print(f"  skip {n}: {e}")
            continue
        proj = "q" if "q_proj" in n else ("k" if "k_proj" in n else ("v" if "v_proj" in n else "o"))
        by_proj[proj].append(rates)
        if "q_proj" in n:
            for h, r in enumerate(rates):
                by_head_global[h].append(r)
        # Per-tensor summary
        print(f"  {n:70s}  {len(rates):5d}   {np.nanmean(rates):.4f} {np.nanstd(rates):.4f} {np.nanmin(rates):.4f} {np.nanmax(rates):.4f}  {np.nanmax(rates) - np.nanmin(rates):.4f}")

    print()
    print(f"{'proj':>4s}  {'tensors':>7s}  {'mean':>7s}  {'std-across-heads':>16s}  {'within-tensor range':>22s}")
    for proj, rates_list in sorted(by_proj.items()):
        all_rates = np.concatenate(rates_list)
        ranges = [np.nanmax(r) - np.nanmin(r) for r in rates_list]
        std_per_t = [np.nanstd(r) for r in rates_list]
        print(f"  {proj:2s}    {len(rates_list):5d}   {np.nanmean(all_rates):.4f}        {np.mean(std_per_t):.4f}              {np.mean(ranges):.4f}")

    # Per-head global view (for q_proj, are the same heads always the highest?)
    if by_head_global:
        print(f"\nPer-head q_proj flip rate (averaged across all q_proj layers):")
        means = np.array([np.nanmean(by_head_global[h]) for h in sorted(by_head_global)])
        print(f"  range across heads: [{means.min():.4f}, {means.max():.4f}]  std: {means.std():.4f}")
        # Which heads are systematically loud?
        sorted_heads = np.argsort(means)
        print(f"  highest-flip heads: {sorted_heads[-5:].tolist()} (rates: {means[sorted_heads[-5:]].round(4).tolist()})")
        print(f"  lowest-flip heads:  {sorted_heads[:5].tolist()} (rates: {means[sorted_heads[:5]].round(4).tolist()})")


if __name__ == "__main__":
    main()
