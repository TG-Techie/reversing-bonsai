# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""PTQ baseline: take Qwen3 base weights, do naive Q1_0_g128 sign-quant
on them, then measure row cosine vs the original.

Why: gives us a calibration constant — pure sign-quant of a
zero-mean weight distribution should land near cos = sqrt(2/pi) ≈ 0.798
(Bussgang's theorem corollary for Gaussian inputs). Bonsai sits at
cos ~= 0.45-0.65 in our H2 measurements. The gap (0.80 → 0.50) is the
"QAT signal" — how much further than thresholding the recipe pushed
the signs.

Crucially: the comparison "Bonsai vs base" measures Bonsai's *learned*
deviation from base. The comparison "PTQ(base) vs base" measures the
*format-induced* deviation. The difference tells us the QAT bit.

Usage:
    uv run python src/ptq_baseline_v2.py models/bonsai/4B/base/ \
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


def ptq_q1_0(W: np.ndarray, group: int = 128) -> np.ndarray:
    """Bonsai-style Q1_0_g128: per-128-block scale = mean(|w|), sign = sign(w)."""
    flat = W.reshape(-1).astype(np.float32)
    n = flat.size
    pad = (-n) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    g = flat.reshape(-1, group)
    s = np.abs(g).mean(axis=-1, keepdims=True)
    sgn = np.where(g >= 0, 1.0, -1.0)
    out = (s * sgn).reshape(-1)[:n]
    return out.reshape(W.shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="dir or .safetensors with Qwen3 base weights")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    base = Path(args.base)
    names = sorted(list_tensor_names(base))
    rows = []

    for n in names:
        if args.filter and args.filter not in n:
            continue
        if not n.endswith(".weight") or "embed" in n or "lm_head" in n:
            continue
        if "norm" in n:
            continue
        W = load_tensor(base, n)
        if W is None or W.ndim != 2 or W.shape[-1] % 128:
            continue
        Wq = ptq_q1_0(W)
        # Row cosine
        a = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        b = Wq / (np.linalg.norm(Wq, axis=1, keepdims=True) + 1e-9)
        cos = (a * b).sum(axis=1)
        # Sign flips (all weights — sign(W) vs sign(Wq); for Wq, sign comes from W)
        # so by construction, flips are 0. But amplitudes differ.
        rows.append((n, W.shape, float(cos.mean()), float(cos.std()), float(cos.min()), float(cos.max())))

    print(f"Compared {len(rows)} tensors against PTQ(base).")
    print(f"{'tensor':70s}  cos_mean   cos_std   cos_min   cos_max")
    by_depth = defaultdict(list)
    for n, sh, m, s, mn, mx in rows:
        d = re.search(r"layers\.(\d+)\.", n)
        if d:
            by_depth[int(d.group(1))].append(m)
        print(f"  {n:70s}  {m:.4f}    {s:.4f}    {mn:.4f}    {mx:.4f}")

    print()
    print(f"{'depth':>6s}  {'mean':>7s}  {'min':>7s}  {'max':>7s}")
    for d in sorted(by_depth):
        v = by_depth[d]
        print(f"  {d:4d}    {np.mean(v):.4f}   {np.min(v):.4f}   {np.max(v):.4f}")

    overall = np.array([m for _, _, m, _, _, _ in rows])
    print(f"\nPTQ(base) row cosine vs base:")
    print(f"  mean = {overall.mean():.4f}")
    print(f"  median = {np.median(overall):.4f}")
    print(f"  range = [{overall.min():.4f}, {overall.max():.4f}]")
    print(f"  prediction sqrt(2/pi) ≈ 0.7979 (Gaussian ideal)")


if __name__ == "__main__":
    main()
