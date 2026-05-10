# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Joint cross-tensor permutation search — the rigorous H4 follow-up.

Per-tensor we already showed (H4) that input-column magnitudes don't
match Qwen3's even under permutation. But a graph-equivalent reorder of
the *residual stream* is a single permutation π applied consistently to
many tensors at once: every Wq/Wk/Wv input column, every Wo output row,
every MLP gate/up input column, every MLP down output row, every layer
norm in-and-out, the embed output rows, and the LM-head input columns.

This script searches for such a π and asks whether it materially
improves the Bonsai-vs-Qwen3 row cosine compared to identity.

We restrict the search to the residual-stream dimension (hidden_size)
because that's the natural "single π" axis. We do NOT touch:
  - the FFN intermediate dimension (a separate permutation would apply
    coherently to gate.rows, up.rows, down.cols), or
  - the head boundary inside attention (constrained by RoPE pair structure).

Method:
  1. Pick a representative tensor whose rows live in the residual stream
     (e.g. the input layernorm output -> Wq input cols). Use Wq's INPUT
     columns directly: Bonsai's Wq[:, j] vs Qwen3's Wq[:, j].
  2. Build a cost matrix C[i, j] = -cosine(Bonsai_col_i, Qwen3_col_j).
  3. Solve linear-sum-assignment to find π minimizing total cost.
  4. Apply π to every input-col-side tensor in Bonsai (Wq, Wk, Wv across
     all layers, plus MLP gate/up).
  5. Compute mean row cosine identity vs Qwen3 base, then mean row cosine
     with π applied. Report the gap.

If π_diag_cos << π_perm_cos, residual permutation is real and we missed
it earlier. Otherwise H4 stands.

Cost: O(n^2) memory for the cost matrix where n = hidden_size. For
Qwen3-1.7B, n = 2048 and the matrix is ~32 MB. Hungarian is O(n^3) and
runs in seconds on CPU via scipy.

Usage:
    uv run python src/joint_permutation_search.py \
        models/unpacked/model.safetensors \
        models/base/model.safetensors \
        [--seed-tensor model.layers.0.self_attn.q_proj.weight]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from compare_unpacked_vs_qwen3 import load_tensor, list_tensor_names


def col_cosine_matrix(b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return (n, n) cosine matrix where entry [i, j] = cos(b[:, i], q[:, j])."""
    bf = b.astype(np.float32)
    qf = q.astype(np.float32)
    bn = bf / (np.linalg.norm(bf, axis=0, keepdims=True) + 1e-12)
    qn = qf / (np.linalg.norm(qf, axis=0, keepdims=True) + 1e-12)
    return bn.T @ qn


def row_cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    an = np.linalg.norm(a, axis=-1) + 1e-12
    bn = np.linalg.norm(b, axis=-1) + 1e-12
    return float((np.einsum("ij,ij->i", a, b) / (an * bn)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bonsai")
    ap.add_argument("base")
    ap.add_argument("--seed-tensor", default="model.layers.0.self_attn.q_proj.weight",
                    help="Tensor whose input-col matching defines π.")
    ap.add_argument("--limit-tensors", type=int, default=12,
                    help="How many input-col-side tensors to evaluate after solving.")
    args = ap.parse_args()

    bp = Path(args.bonsai)
    qp = Path(args.base)

    # Step 1: solve for π using the seed tensor's input columns
    print(f"[seed] {args.seed_tensor}")
    seed_b = load_tensor(bp, args.seed_tensor)
    seed_q = load_tensor(qp, args.seed_tensor)
    if seed_b is None or seed_q is None:
        print(f"missing seed tensor on one side", file=sys.stderr)
        sys.exit(2)
    if seed_b.shape != seed_q.shape:
        print(f"seed shape mismatch {seed_b.shape} vs {seed_q.shape}", file=sys.stderr)
        sys.exit(2)
    print(f"  shape={seed_b.shape}  (out, in_residual)={seed_b.shape}")

    # Cosine matrix (n, n) where n = hidden / residual-stream dim
    print(f"[search] building col cosine matrix ({seed_b.shape[1]} x {seed_b.shape[1]})")
    C_cos = col_cosine_matrix(seed_b, seed_q)
    cost = -C_cos  # minimize negative cosine
    print(f"[search] running linear_sum_assignment...")
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind = [0,1,...n-1]; col_ind[i] = which Qwen3 column matches Bonsai col i.
    pi = col_ind  # Bonsai column i pairs with Qwen3 column pi[i]
    n_identity_match = int((pi == np.arange(pi.size)).sum())
    print(f"[search] perm has {n_identity_match}/{pi.size} fixed points")
    # Diagnostic: per-pair cosine under π vs identity
    diag_cos_identity = float(np.diag(C_cos).mean())
    diag_cos_perm = float(C_cos[np.arange(pi.size), pi].mean())
    print(f"[seed] mean col-cosine identity={diag_cos_identity:.4f}  perm={diag_cos_perm:.4f}")

    # Step 2: apply π and re-evaluate row cosines on a battery of input-col-side
    # tensors (the ones that, under graph equivalence with a residual-stream
    # permutation, would all benefit from π applied to their input columns).
    print()
    print("== applying π to Bonsai input columns and comparing to base ==")

    # Build a list of input-col-side weight names
    common = sorted(set(list_tensor_names(bp)) & set(list_tensor_names(qp)))
    candidates = []
    for name in common:
        # input-col-side: q_proj, k_proj, v_proj (input from residual);
        # gate_proj, up_proj (input from residual). All 2D.
        if not (name.endswith("self_attn.q_proj.weight") or
                name.endswith("self_attn.k_proj.weight") or
                name.endswith("self_attn.v_proj.weight") or
                name.endswith("mlp.gate_proj.weight") or
                name.endswith("mlp.up_proj.weight")):
            continue
        candidates.append(name)
    candidates = candidates[: args.limit_tensors]

    rows = []
    for name in candidates:
        bw = load_tensor(bp, name)
        qw = load_tensor(qp, name)
        if bw is None or qw is None or bw.shape != qw.shape or bw.ndim != 2:
            continue
        cos_id = row_cos_mean(bw, qw)
        # Apply π to the INPUT COL dim of bw (axis 1)
        bw_pi = bw[:, pi]
        cos_pi = row_cos_mean(bw_pi, qw)
        rows.append((name, cos_id, cos_pi))
        print(f"  {name:60s}  cos_id={cos_id:.4f}  cos_pi={cos_pi:.4f}  Δ={cos_pi - cos_id:+.4f}")

    if rows:
        cos_id_arr = np.array([r[1] for r in rows])
        cos_pi_arr = np.array([r[2] for r in rows])
        delta = cos_pi_arr - cos_id_arr
        print()
        print("== AGGREGATE ==")
        print(f"  identity row cosine:  mean={cos_id_arr.mean():.4f}  min={cos_id_arr.min():.4f}  max={cos_id_arr.max():.4f}")
        print(f"  π-applied row cosine: mean={cos_pi_arr.mean():.4f}  min={cos_pi_arr.min():.4f}  max={cos_pi_arr.max():.4f}")
        print(f"  Δ (π − identity):     mean={delta.mean():+.4f}  min={delta.min():+.4f}  max={delta.max():+.4f}")
        print()
        print("Reading guide:")
        print("  Δ ≈ 0  → no consistent residual permutation; the seed-tensor π doesn't")
        print("           generalize. H4 caveat falls cleanly: residual not permuted.")
        print("  Δ > 0.05 (and especially Δ > 0.2) → a single residual permutation")
        print("           materially improves alignment; H2/H4 should be re-opened.")


if __name__ == "__main__":
    main()
