# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Per-tensor sign-disagreement count between Bonsai-unpacked and Qwen3-base.

H2 reports row cosine. This adds the more direct picture: across each weight
tensor, what *fraction of weights* changed sign between Qwen3 and Bonsai?

Usage:
    uv run python src/sign_disagreement.py \
        models/bonsai/<size>/unpacked/  models/bonsai/<size>/base/ \
        [--filter SUBSTR]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unpacked")
    ap.add_argument("base")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    unp = Path(args.unpacked)
    base = Path(args.base)
    names = sorted(set(list_tensor_names(unp)) & set(list_tensor_names(base)))

    rows = []
    for n in names:
        if args.filter and args.filter not in n:
            continue
        # Skip 1D / size mismatches; we want matrix-heavy tensors
        if not n.endswith(".weight"):
            continue
        if "embed" in n or "lm_head" in n:
            continue  # treat separately if needed
        u = load_tensor(unp, n)
        b = load_tensor(base, n)
        if u is None or b is None or u.shape != b.shape or u.ndim != 2:
            continue
        # Sign disagreement on nonzero entries
        nz = (np.abs(u) > 0) & (np.abs(b) > 0)
        if nz.sum() == 0:
            continue
        flips = (np.sign(u[nz]) != np.sign(b[nz])).sum() / nz.sum()
        rows.append((n, u.shape, float(flips)))

    if not rows:
        print("No tensors compared.")
        return

    print(f"Compared {len(rows)} tensors.")
    print(f"{'tensor':70s}  flip_frac")
    # Group by layer + projection type for a depth view
    by_depth = defaultdict(list)
    for n, sh, f in rows:
        m = re.match(r"model\.layers\.(\d+)\.", n)
        if m:
            by_depth[int(m.group(1))].append((n, f))
        print(f"  {n:70s}  {f:.4f}")

    print()
    print(f"{'depth':>6s}  {'mean':>7s}  {'min':>7s}  {'max':>7s}  n")
    for d in sorted(by_depth):
        fs = [f for _, f in by_depth[d]]
        print(f"  {d:4d}    {np.mean(fs):.4f}   {np.min(fs):.4f}   {np.max(fs):.4f}   {len(fs)}")

    overall = np.array([f for _, _, f in rows])
    print(f"\nOVERALL: mean={overall.mean():.4f}  median={np.median(overall):.4f}  min={overall.min():.4f}  max={overall.max():.4f}")
    print(f"Random baseline (50/50 signs): 0.5000")
    print(f"Pure Gaussian sign-quant prediction (cos = sqrt(2/pi)≈0.798): flip ≈ (1-0.798)/2 ≈ 0.1010")


if __name__ == "__main__":
    main()
