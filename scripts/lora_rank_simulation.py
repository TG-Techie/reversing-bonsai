"""Test what rank-r LoRA delta added to teacher reproduces the
L1 ffn_gate over-dispersion of ~10 at 8B.

Sweep r ∈ {1, 2, 4, 8, 16, 32, 64} and find the (r, sigma) that
matches both Bonsai's L1 ffn_gate flip rate (~0.34) AND its
over-dispersion (~10.27).

If a small-r LoRA can reproduce both, that's evidence for an
MLP-only small-rank LoRA component at L1-3.
"""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import safe_open as safe_open_pt

BASE_DIR = ROOT / "models/bonsai/8B/base"
key_to_shard = {}
base_shards = sorted(BASE_DIR.glob("*.safetensors"))
for s in base_shards:
    with safe_open_pt(str(s), framework="pt") as f:
        for k in f.keys():
            key_to_shard[k] = s


def load_base(name):
    s = key_to_shard[name]
    with safe_open_pt(str(s), framework="pt") as f:
        return f.get_tensor(name).to(torch.float32).numpy()


import os
LAYER = int(os.environ.get("LAYER", "1"))
TARGET_P = float(os.environ.get("TARGET_P", "0.3449"))
# Load teacher gate at LAYER
W_t = load_base(f"model.layers.{LAYER}.mlp.gate_proj.weight")
print(f"Teacher L{LAYER} ffn_gate shape: {W_t.shape}")
out_dim, in_dim = W_t.shape
print(f"Teacher mean(|w|): {np.abs(W_t).mean():.6f}")

# Reshape to blocks: (out * nblk, 128)
flat = W_t.reshape(-1, 128)


def overdisp_for_perturbed(W_pert, W_t):
    """Compute over-dispersion of sign-flip count under given perturbation."""
    sign_t = np.sign(W_t)
    sign_p = np.sign(W_pert)
    nz = sign_t != 0
    flips_per_block = (sign_t != sign_p).reshape(-1, 128).sum(axis=1)
    p = flips_per_block.mean() / 128.0
    obs_var = flips_per_block.var()
    exp_var = 128.0 * p * (1.0 - p)
    return p, float(obs_var / exp_var) if exp_var > 0 else float("nan")


target_p = TARGET_P
print(f"\nTarget flip rate: p={target_p}")

# Try LoRA rank-r perturbations with various sigmas
# We want sigma chosen to land flip rate ~0.34, then check overdisp
print(f"\nSweep over rank-r LoRA: {'r':>3} {'sigma':>6} {'flip_p':>8} {'overdisp':>9}")
rng = np.random.default_rng(42)
for r in [1, 2, 4, 8, 16, 32, 64, 128]:
    # Rank-r LoRA: W' = W + sigma * U V^T where U ~ N(0,1) (out, r), V ~ N(0,1) (in, r)
    # Tune sigma to give flip rate ~0.34
    U = rng.normal(0, 1, size=(out_dim, r)).astype(np.float32)
    V = rng.normal(0, 1, size=(r, in_dim)).astype(np.float32)
    delta_unit = U @ V  # (out_dim, in_dim)
    delta_unit_std = delta_unit.std()
    # Calibrate sigma via binary search to get target flip rate
    lo, hi = 0.0, 0.5
    for _ in range(20):
        sigma = (lo + hi) / 2
        W_pert = W_t + (sigma / max(delta_unit_std, 1e-9)) * delta_unit
        p, _ = overdisp_for_perturbed(W_pert, W_t)
        if p < target_p:
            lo = sigma
        else:
            hi = sigma
    # Final measurement
    W_pert = W_t + (sigma / max(delta_unit_std, 1e-9)) * delta_unit
    p, od = overdisp_for_perturbed(W_pert, W_t)
    print(f" {r:>3} {sigma:>6.3f} {p:>8.4f} {od:>9.3f}")
