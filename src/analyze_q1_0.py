"""Analyze Q1_0 tensors in a Bonsai GGUF file.

For each Q1_0 tensor we compute statistics that bear on the user's question
("are the weights sorted? are they permuted before grouping?").

Per-block diagnostics:
    - signs balance (count of +1 vs -1; balanced groups => mean ~ 0 of original)
    - sign run-length pattern -> if blocks are sorted by sign, runs collapse
      to <=2; random signs give Geom(0.5) run lengths
    - sign autocorrelation at lag 1
    - "monotonic in sign" indicator (signs all -1 then all +1)

Per-tensor diagnostics:
    - distribution of scales d_g (mean, std, hist)
    - sortedness of scales over groups (Kendall-tau-like measure: fraction of
      adjacent pairs in non-decreasing order)
    - global histogram of signs
    - exact size match vs ggml_row_size

Usage:
    uv run python src/analyze_q1_0.py <path.gguf> [--filter SUBSTR] [--top N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import QK1_0, parse_q1_0, BLOCK_BYTES  # noqa: E402


def sign_runs_summary(signs: np.ndarray) -> dict:
    """signs shape (nblocks, 128) ∈ {-1,+1}. Returns aggregate stats."""
    nb = signs.shape[0]
    # Number of sign transitions per block (0 means all same sign;
    # 1 means perfectly sorted in two contiguous halves).
    transitions = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    # +1 / -1 counts per block
    pos = np.sum(signs == 1, axis=1)
    neg = QK1_0 - pos
    # autocorr lag-1 of signs (per block, then mean): expected ~0 for random
    s = signs.astype(np.float32)
    ac1 = np.mean((s[:, :-1] * s[:, 1:]).astype(np.float32), axis=1)
    return {
        "nblocks": int(nb),
        "transitions_mean": float(transitions.mean()),
        "transitions_std":  float(transitions.std()),
        "transitions_min":  int(transitions.min()),
        "transitions_max":  int(transitions.max()),
        "frac_blocks_sorted_<=1_transition": float((transitions <= 1).mean()),
        "frac_blocks_all_same_sign": float((transitions == 0).mean()),
        "pos_count_mean": float(pos.mean()),
        "pos_count_std":  float(pos.std()),
        "neg_count_mean": float(neg.mean()),
        "ac_lag1_mean": float(ac1.mean()),
        "ac_lag1_std":  float(ac1.std()),
    }


def scale_summary(scales: np.ndarray) -> dict:
    """scales shape (nblocks,) float32."""
    nb = scales.shape[0]
    diffs = scales[1:] - scales[:-1]
    return {
        "scale_mean": float(scales.mean()),
        "scale_std":  float(scales.std()),
        "scale_min":  float(scales.min()),
        "scale_max":  float(scales.max()),
        "frac_pairs_nondecreasing": float((diffs >= 0).mean()) if nb > 1 else 1.0,
        "frac_pairs_nonincreasing": float((diffs <= 0).mean()) if nb > 1 else 1.0,
        # number of distinct unique scales (FP16 quantized)
        "distinct_scales": int(np.unique(scales.astype(np.float16)).size),
    }


def column_balance_summary(signs: np.ndarray, n_cols: int, k: int = 128) -> dict:
    """If the tensor is 2D with n_cols, blocks of 128 are along the last dim.

    Each row contains n_cols/k blocks. Within a row, are blocks ordered by
    abs-sum or some other monotone? We compute:
      * for each row, fraction of monotone (sorted) sequences of block scales
      * (already covered above globally; here we condition per-row)
    """
    nb_total = signs.shape[0]
    if n_cols % k:
        return {}
    blocks_per_row = n_cols // k
    if nb_total % blocks_per_row:
        return {}
    nrows = nb_total // blocks_per_row
    return {"rows": int(nrows), "blocks_per_row": int(blocks_per_row)}


def analyze(path: str, filt: str | None = None, top: int = 10) -> None:
    r = GGUFReader(path, "r")
    print(f"== {path} ==")
    print(f"  tensors: {len(r.tensors)}\n")

    # Find Q1_0 tensors
    q1 = [t for t in r.tensors
          if t.tensor_type == GGMLQuantizationType.Q1_0
          and (filt is None or filt in t.name)]
    print(f"  Q1_0 tensors (after filter): {len(q1)}\n")

    summary = []
    for i, t in enumerate(q1[: top if top > 0 else None]):
        n_elems = int(np.prod(t.shape))
        nb = n_elems // QK1_0
        if n_elems % QK1_0:
            print(f"!! tensor {t.name} not multiple of {QK1_0} (n={n_elems})")
            continue

        # Read raw bytes for this tensor.
        # GGUFReader gives us t.data as a uint8 numpy array of n_bytes bytes.
        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        if len(raw) != nb * BLOCK_BYTES:
            print(f"!! size mismatch for {t.name}: raw={len(raw)} expected={nb*BLOCK_BYTES}")
            continue

        scales, signs = parse_q1_0(raw, n_elems)
        ss = sign_runs_summary(signs)
        cs = scale_summary(scales)

        n_cols = int(t.shape[0])  # GGUF shape is reversed from numpy: shape[0] is fastest dim
        cb = column_balance_summary(signs, n_cols)

        print(f"-- [{i}] {t.name}  shape={list(t.shape)}  blocks={nb}  ({nb*BLOCK_BYTES} bytes)")
        print(f"     signs:   trans μ={ss['transitions_mean']:.2f} σ={ss['transitions_std']:.2f}  "
              f"[{ss['transitions_min']},{ss['transitions_max']}];  "
              f"all-same={ss['frac_blocks_all_same_sign']*100:.2f}%  "
              f"<=1trans={ss['frac_blocks_sorted_<=1_transition']*100:.2f}%")
        print(f"     pos/blk: μ={ss['pos_count_mean']:.2f} σ={ss['pos_count_std']:.2f}; "
              f"ac1={ss['ac_lag1_mean']:+.4f}±{ss['ac_lag1_std']:.4f}")
        print(f"     scales:  μ={cs['scale_mean']:.4g} σ={cs['scale_std']:.4g}  "
              f"[{cs['scale_min']:.4g},{cs['scale_max']:.4g}];  "
              f"distinct(fp16)={cs['distinct_scales']}/{nb}")
        print(f"     scale order: nondec={cs['frac_pairs_nondecreasing']*100:.2f}%  "
              f"noninc={cs['frac_pairs_nonincreasing']*100:.2f}%")
        if cb:
            print(f"     layout:  {cb['rows']} rows × {cb['blocks_per_row']} blocks/row")

        summary.append({
            "name": t.name, "shape": list(t.shape),
            "signs": ss, "scales": cs, "cb": cb,
        })

    # global aggregate
    if summary:
        print("\n== GLOBAL Q1_0 AGGREGATE ==")
        keys = ["transitions_mean", "frac_blocks_all_same_sign",
                "frac_blocks_sorted_<=1_transition", "ac_lag1_mean",
                "pos_count_mean"]
        for k in keys:
            vals = [s["signs"][k] for s in summary]
            print(f"  {k:40s} mean={np.mean(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")
        for k in ["scale_mean", "scale_std", "frac_pairs_nondecreasing"]:
            vals = [s["scales"][k] for s in summary]
            print(f"  {k:40s} mean={np.mean(vals):.4g}  min={np.min(vals):.4g}  max={np.max(vals):.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--top", type=int, default=10,
                    help="Limit per-tensor output to first N (0 = all)")
    args = ap.parse_args()
    analyze(args.path, args.filter, args.top)


if __name__ == "__main__":
    main()
