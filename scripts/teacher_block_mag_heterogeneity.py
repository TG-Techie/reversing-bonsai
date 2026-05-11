"""Check whether L1-3 ffn_gate's INTRINSIC teacher weight distribution
has higher within-block magnitude heterogeneity than L0/baseline.

If yes, the "L1-3 over-dispersion spike" might be substantially
explained by teacher structure, not by recipe choices — a 4th
potential confound on top of teacher-sign-blockstruct (which we
already cleared in 40_*).

Per-block: fraction of positions with |w| < 0.5 * mean(|w|_block).
"Near-zero positions" — these flip at near-random rate under any
perturbation. The cross-block variance of this fraction quantifies
how heterogeneous teacher's within-block magnitude distribution is.
"""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import safe_open as safe_open_pt

BASE_DIR = ROOT / "models/bonsai/8B/base"
key_to_shard = {}
for s in sorted(BASE_DIR.glob("*.safetensors")):
    with safe_open_pt(str(s), framework="pt") as f:
        for k in f.keys():
            key_to_shard[k] = s


def load_base(name):
    s = key_to_shard[name]
    with safe_open_pt(str(s), framework="pt") as f:
        return f.get_tensor(name).to(torch.float32).numpy()


def block_mag_heterogeneity(name):
    W = load_base(name)
    flat = W.reshape(-1, 128)
    block_mean_abs = np.abs(flat).mean(axis=1)  # (nblk,)
    # Within-block CV of |w| (a measure of how spread the magnitudes are)
    block_std_abs = np.abs(flat).std(axis=1)
    cv_per_block = block_std_abs / np.maximum(block_mean_abs, 1e-9)
    # Cross-block stats
    return {
        "global_mean_abs": float(np.abs(flat).mean()),
        "block_mean_abs_mean": float(block_mean_abs.mean()),
        "block_mean_abs_std": float(block_mean_abs.std()),
        "block_mean_abs_cv": float(block_mean_abs.std() / max(block_mean_abs.mean(), 1e-9)),
        "within_block_cv_mean": float(cv_per_block.mean()),
        "within_block_cv_std": float(cv_per_block.std()),
        # Fraction of "small" positions per block (|w| < 0.5 * block_mean_abs)
        "near_zero_frac_mean": float((np.abs(flat) < 0.5 * block_mean_abs[:, None]).mean()),
        "near_zero_frac_std": float((np.abs(flat) < 0.5 * block_mean_abs[:, None]).mean(axis=1).std()),
    }


print(f"{'tensor':<28} {'g_meanabs':>10} {'blk_cv':>8} {'within_cv':>10} {'within_cv_std':>14} {'nzfrac_std':>11}")
for L in [0, 1, 2, 3, 9, 18, 27, 35]:
    for kind in ["mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj"]:
        name = f"model.layers.{L}.{kind}.weight"
        try:
            r = block_mag_heterogeneity(name)
            label = f"L{L} {kind.split('.')[-1]}"
            print(f"{label:<28} {r['global_mean_abs']:>10.6f} {r['block_mean_abs_cv']:>8.3f} {r['within_block_cv_mean']:>10.3f} {r['within_block_cv_std']:>14.3f} {r['near_zero_frac_std']:>11.4f}")
        except Exception as e:
            print(f"{kind} L{L}: ERR {e}")
