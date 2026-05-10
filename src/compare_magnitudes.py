# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Compare |w| distributions between Bonsai and Qwen3 base, at three cuts.

We've already shown that Bonsai weights live on a binary lattice {±s_g} per
128-block. The interesting follow-up: did Bonsai keep Qwen3's magnitude
profile, or did QAT re-learn magnitudes too?

For each shared weight matrix we report three correlations and one
distributional comparison:

  * Per-128-block (the quantization unit, blocks along the fast/`in` dim):
      Bonsai s_g vs base group's mean(|w|), median(|w|), max(|w|), std(|w|).
      A correlation near 1.0 with mean(|w|) means "Bonsai used Qwen3's group
      magnitude as the scale"; near 0 means QAT picked unrelated magnitudes.

  * Per-column (fixed input feature, all outputs):
      column-wise mean |w| in Bonsai vs base. Pearson correlation.

  * Per-row (fixed output feature, all inputs):
      row-wise mean |w| in Bonsai vs base. Pearson correlation.

  * Histogram overlap: K-S statistic between {|w_bonsai|} and {|w_base|}
    sampled from the matrix; tells us if the global |w| distribution shifted.

Usage:
    uv run python src/compare_magnitudes.py \
        models/q1/Bonsai-1.7B-Q1_0.gguf \
        models/unpacked/model.safetensors \
        models/base/model-00001-of-00002.safetensors \
        [--filter SUBSTR] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import QK1_0, parse_q1_0
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates
from compare_unpacked_vs_qwen3 import load_tensor as load_st_or_gguf


def load_q1_block_scales(reader: GGUFReader, name: str) -> tuple[np.ndarray, list[int]]:
    """Return per-block FP16 scales (1D float32) and HF-orientation shape."""
    for t in reader.tensors:
        if t.name != name:
            continue
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            raise TypeError(name)
        gguf_shape = list(t.shape)
        hf_shape = list(reversed(gguf_shape))
        n_elems = int(np.prod(gguf_shape))
        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        scales, _signs = parse_q1_0(raw, n_elems)
        return scales, hf_shape
    raise KeyError(name)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return float("nan")
    return float((a @ b) / den)


def ks_statistic(x: np.ndarray, y: np.ndarray, n_samples: int = 100_000) -> float:
    """Two-sample Kolmogorov-Smirnov, capped sample size for speed."""
    rng = np.random.default_rng(0)
    if x.size > n_samples:
        x = rng.choice(x, size=n_samples, replace=False)
    if y.size > n_samples:
        y = rng.choice(y, size=n_samples, replace=False)
    xs = np.sort(x)
    ys = np.sort(y)
    combined = np.concatenate([xs, ys])
    cdf_x = np.searchsorted(xs, combined, side="right") / xs.size
    cdf_y = np.searchsorted(ys, combined, side="right") / ys.size
    return float(np.abs(cdf_x - cdf_y).max())


def per_block_stats(base_arr: np.ndarray, group: int = QK1_0) -> dict[str, np.ndarray]:
    """Group base_arr along its FAST dim (= last dim in HF orientation)."""
    flat = base_arr.reshape(-1)
    n = flat.size
    if n % group:
        raise ValueError(f"size {n} not multiple of {group}")
    g = np.abs(flat.reshape(-1, group).astype(np.float32))
    return {
        "mean":   g.mean(axis=1),
        "median": np.median(g, axis=1),
        "max":    g.max(axis=1),
        "std":    g.std(axis=1),
    }


def compare_one(name: str, q1: GGUFReader, unp: Path, base: Path) -> dict:
    out: dict = {"gguf_name": name}
    try:
        scales, hf_shape = load_q1_block_scales(q1, name)
    except (KeyError, TypeError) as e:
        return {"gguf_name": name, "error": str(e)}

    cand = gguf_to_hf_candidates(name)
    out["hf_candidates"] = cand
    base_arr = None
    found = None
    for c in cand:
        base_arr = load_st_or_gguf(base, c)
        if base_arr is not None:
            found = c
            break
    if base_arr is None:
        return {"gguf_name": name, "error": f"base tensor not found ({cand})"}
    out["hf_name"] = found
    out["shape_q1_hf"] = hf_shape
    out["shape_base"] = list(base_arr.shape)

    # Reshape base into HF-orientation matching q1's hf_shape if shapes match.
    if tuple(base_arr.shape) != tuple(hf_shape):
        out["error"] = f"shape mismatch q1={hf_shape} base={base_arr.shape}"
        return out

    # Per-block stats of base
    bs = per_block_stats(base_arr, QK1_0)
    out["n_blocks"] = int(scales.size)

    # Correlations of Bonsai s_g vs base block stats
    out["corr_s_vs_base_mean"]   = pearson(scales, bs["mean"])
    out["corr_s_vs_base_median"] = pearson(scales, bs["median"])
    out["corr_s_vs_base_max"]    = pearson(scales, bs["max"])
    out["corr_s_vs_base_std"]    = pearson(scales, bs["std"])

    # Ratio: average s_g / base group mean -> close to 1.0 means scales are
    # the same magnitude as base group means; >1 means Bonsai is "louder",
    # <1 means quieter.
    eps = 1e-12
    ratio = scales / (bs["mean"] + eps)
    out["s_over_base_mean_median"] = float(np.median(ratio))
    out["s_over_base_mean_p10p90"] = (float(np.percentile(ratio, 10)),
                                       float(np.percentile(ratio, 90)))

    # Load Bonsai unpacked too, to anchor "Bonsai per-block mean(|w|) == s_g"
    # and so we have a direct |w| sample on the Bonsai side.
    bonsai_arr = load_st_or_gguf(unp, found)
    if bonsai_arr is not None and tuple(bonsai_arr.shape) == tuple(hf_shape):
        # Per-column / per-row mean |w|
        # In HF: shape = (out, in). axis=0 -> per-input-column; axis=1 -> per-row.
        bons_abs = np.abs(bonsai_arr.astype(np.float32))
        base_abs = np.abs(base_arr.astype(np.float32))
        out["corr_col_meanabs"] = pearson(bons_abs.mean(axis=0), base_abs.mean(axis=0))
        out["corr_row_meanabs"] = pearson(bons_abs.mean(axis=1), base_abs.mean(axis=1))
        # Global magnitude shape comparison
        out["ks_abs"] = ks_statistic(bons_abs.ravel(), base_abs.ravel())
        out["abs_mean_bonsai"] = float(bons_abs.mean())
        out["abs_mean_base"] = float(base_abs.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="Bonsai-*-Q1_0.gguf")
    ap.add_argument("unpacked", help="Bonsai unpacked safetensors (single file)")
    ap.add_argument("base", help="Qwen3 base safetensors (single shard or single file)")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    reader = GGUFReader(args.gguf, "r")
    unp = Path(args.unpacked)
    base = Path(args.base)
    q1_tensors = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.Q1_0]
    print(f"== {len(q1_tensors)} Q1_0 tensors")

    # Aggregate
    all_corr_mean = []
    all_corr_median = []
    all_corr_max = []
    all_corr_std = []
    all_corr_col = []
    all_corr_row = []
    all_ks = []
    all_ratio = []

    seen = 0
    for t in q1_tensors:
        if args.filter and args.filter not in t.name:
            continue
        r = compare_one(t.name, reader, unp, base)
        seen += 1
        if "error" in r:
            print(f"\n-- {r['gguf_name']}\n   error: {r['error']}")
            continue
        line = (
            f"\n-- {r['gguf_name']} -> {r['hf_name']}  shape={r['shape_q1_hf']}\n"
            f"   per-block (n={r['n_blocks']}): corr s_g vs base..\n"
            f"     mean(|w|)   = {r['corr_s_vs_base_mean']:+.4f}\n"
            f"     median(|w|) = {r['corr_s_vs_base_median']:+.4f}\n"
            f"     max(|w|)    = {r['corr_s_vs_base_max']:+.4f}\n"
            f"     std(|w|)    = {r['corr_s_vs_base_std']:+.4f}\n"
            f"   ratio s_g / base_mean: median={r['s_over_base_mean_median']:.3f}"
            f"  p10..p90={r['s_over_base_mean_p10p90'][0]:.3f}..{r['s_over_base_mean_p10p90'][1]:.3f}"
        )
        if "corr_col_meanabs" in r:
            line += (
                f"\n   per-col   corr mean|w|: {r['corr_col_meanabs']:+.4f}"
                f"\n   per-row   corr mean|w|: {r['corr_row_meanabs']:+.4f}"
                f"\n   global KS(|w|):          {r['ks_abs']:.4f}"
                f"  mean|w|: bonsai={r['abs_mean_bonsai']:.4f} base={r['abs_mean_base']:.4f}"
            )
            all_corr_col.append(r["corr_col_meanabs"])
            all_corr_row.append(r["corr_row_meanabs"])
            all_ks.append(r["ks_abs"])
        print(line)

        all_corr_mean.append(r["corr_s_vs_base_mean"])
        all_corr_median.append(r["corr_s_vs_base_median"])
        all_corr_max.append(r["corr_s_vs_base_max"])
        all_corr_std.append(r["corr_s_vs_base_std"])
        all_ratio.append(r["s_over_base_mean_median"])

        if args.limit and seen >= args.limit:
            break

    if all_corr_mean:
        print("\n== AGGREGATE (across compared tensors) ==")
        for label, vals in [
            ("corr s_g vs base mean(|w|)",   all_corr_mean),
            ("corr s_g vs base median(|w|)", all_corr_median),
            ("corr s_g vs base max(|w|)",    all_corr_max),
            ("corr s_g vs base std(|w|)",    all_corr_std),
            ("corr per-col mean|w|",         all_corr_col),
            ("corr per-row mean|w|",         all_corr_row),
            ("KS(|w_bonsai|, |w_base|)",     all_ks),
            ("ratio s_g / base_mean (median per tensor)", all_ratio),
        ]:
            if not vals:
                continue
            v = np.asarray(vals, dtype=np.float64)
            print(f"  {label:48s}  mean={v.mean():+.4f}  min={v.min():+.4f}  max={v.max():+.4f}")


if __name__ == "__main__":
    main()
