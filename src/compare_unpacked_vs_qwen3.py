# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Compare Bonsai-{size}-unpacked (FP16) against Qwen3-{size} base (FP16).

The user's hypothesis: Bonsai's Q1_0 quant unpacks losslessly into the FP16
representation of the base Qwen3 model. Verifying that requires:

  1) Each Bonsai weight tensor's value lattice is exactly {±s_g} per 128-block
     (i.e. binary lattice). Trivially true if Bonsai-unpacked was built by
     dequantizing Q1_0.
  2) The Bonsai unpacked weight at position (i, j) == Qwen3 weight at
     position (π(i), σ(j)) for some permutations π, σ. If π = σ = identity
     and equality holds element-wise, Bonsai is just a sign-quantized Qwen3.
     If π,σ are nontrivial, Bonsai = Qwen3 with channels reordered prior to
     1-bit quantization (so each 128-block has carefully-chosen members).
  3) The 128-element groups are themselves "sorted" in some sense: e.g. each
     group's elements all share the same magnitude (as required for binary
     reconstruction), or scales are monotone across groups.

This script reports per-tensor:
   - shape match
   - value lattice: number of distinct |w|s per 128-group (should be 1 if
     the tensor lives on a binary lattice)
   - identity-permutation L2 / cosine similarity to Qwen3 base
   - row-permutation that maximizes cosine similarity (greedy, last-axis)
   - column-permutation candidate via abs-sorted alignment

Inputs may be safetensors or GGUF F16. Auto-detects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_tensor(path: Path, name: str) -> np.ndarray | None:
    """Load a tensor from safetensors or GGUF, return float32 numpy array."""
    sfx = path.suffix.lower()
    if sfx == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="numpy") as f:
            if name not in f.keys():
                return None
            return f.get_tensor(name).astype(np.float32)
    if sfx == ".gguf":
        from gguf import GGUFReader
        r = GGUFReader(str(path), "r")
        for t in r.tensors:
            if t.name == name:
                # GGUF F16 tensors come back as f16; non-F16 we won't try here.
                if str(t.tensor_type.name) in ("F16", "F32", "BF16"):
                    arr = np.asarray(t.data)
                    return arr.astype(np.float32)
                return None
    raise ValueError(f"Unsupported suffix: {path}")


def list_tensor_names(path: Path) -> list[str]:
    sfx = path.suffix.lower()
    if sfx == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="numpy") as f:
            return list(f.keys())
    if sfx == ".gguf":
        from gguf import GGUFReader
        r = GGUFReader(str(path), "r")
        return [t.name for t in r.tensors]
    return []


def lattice_stats(w: np.ndarray, group: int = 128) -> dict:
    """How close is `w`'s last dim to a binary lattice (each group has 1 magnitude)?"""
    flat = w.reshape(-1)
    if flat.size % group:
        return {"error": "size not multiple of group"}
    g = flat.reshape(-1, group)
    abs_g = np.abs(g)
    # Per-group: distinct magnitudes (rounded a touch to absorb fp16 noise)
    rounded = np.round(abs_g.astype(np.float32) * 1e6) / 1e6
    distinct = np.array([np.unique(row).size for row in rounded])
    # What fraction of groups have exactly 1 distinct magnitude (binary lattice)?
    return {
        "groups": int(g.shape[0]),
        "frac_binary_lattice": float((distinct == 1).mean()),
        "max_distinct_per_group": int(distinct.max()),
        "median_distinct_per_group": float(np.median(distinct)),
        "mean_abs": float(abs_g.mean()),
        "frac_zero": float((flat == 0).mean()),
    }


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a, b shape (rows, cols). Returns (rows,)."""
    an = np.linalg.norm(a, axis=-1) + 1e-12
    bn = np.linalg.norm(b, axis=-1) + 1e-12
    return np.einsum("ij,ij->i", a, b) / (an * bn)


def best_row_permutation(bonsai: np.ndarray, base: np.ndarray, max_rows: int = 4096) -> dict:
    """Find a row permutation π such that bonsai[π[i]] best matches base[i].

    Greedy nearest-neighbor; uses cosine similarity. For huge matrices we sample.
    """
    rows = bonsai.shape[0]
    if rows > max_rows:
        idx = np.random.default_rng(0).choice(rows, size=max_rows, replace=False)
        bonsai_s = bonsai[idx]
        base_s = base[idx]
    else:
        bonsai_s = bonsai
        base_s = base

    # Normalize
    a = bonsai_s / (np.linalg.norm(bonsai_s, axis=-1, keepdims=True) + 1e-12)
    b = base_s / (np.linalg.norm(base_s, axis=-1, keepdims=True) + 1e-12)
    # Greedy: for each base row, find best bonsai row (without replacement)
    sim = b @ a.T  # (rows, rows)
    n = sim.shape[0]
    used = np.zeros(n, dtype=bool)
    perm = np.full(n, -1, dtype=np.int64)
    order = np.argsort(-sim.max(axis=1))  # rows with strongest unique match first
    diag_id = float(np.einsum("ij,ij->i", a, b).mean())
    for i in order:
        candidates = np.argsort(-sim[i])
        for c in candidates:
            if not used[c]:
                perm[i] = c
                used[c] = True
                break
    matched = float(sim[np.arange(n), perm].mean())
    return {
        "rows_compared": int(n),
        "identity_diag_cos": diag_id,
        "best_perm_cos": matched,
        "perm_is_identity": bool((perm == np.arange(n)).all()),
        "perm_first8": perm[:8].tolist(),
    }


def compare_one(name: str, bonsai_path: Path, base_path: Path) -> dict:
    bw = load_tensor(bonsai_path, name)
    qw = load_tensor(base_path, name)
    if bw is None or qw is None:
        return {"name": name, "error": "missing"}
    if bw.shape != qw.shape:
        return {"name": name, "error": f"shape {bw.shape} vs {qw.shape}"}
    if bw.ndim != 2:
        return {"name": name, "shape": list(bw.shape),
                "lattice": lattice_stats(bw)}

    # Identity comparison
    diff = bw - qw
    rmse = float(np.sqrt(np.mean(diff**2)))
    cos_id = cosine_per_row(bw, qw).mean()

    out = {
        "name": name, "shape": list(bw.shape),
        "rmse_identity": rmse,
        "cos_identity_mean": float(cos_id),
        "lattice_bonsai": lattice_stats(bw),
        "lattice_base":   lattice_stats(qw),
    }

    # Row-permutation search (only if shapes are tractable)
    if bw.shape[0] <= 32768:
        out["row_perm"] = best_row_permutation(bw, qw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bonsai", help="Path to Bonsai-unpacked safetensors or GGUF F16")
    ap.add_argument("base",   help="Path to Qwen3 base safetensors or GGUF F16")
    ap.add_argument("--filter", default=None,
                    help="Only compare tensors containing this substring")
    ap.add_argument("--names", action="store_true",
                    help="Just list common tensor names and exit")
    args = ap.parse_args()

    bp, qp = Path(args.bonsai), Path(args.base)
    bn = list_tensor_names(bp)
    qn = list_tensor_names(qp)
    common = [n for n in bn if n in set(qn)]
    print(f"bonsai tensors: {len(bn)}; base tensors: {len(qn)}; common: {len(common)}")
    if args.names:
        for n in common[:40]:
            print(" ", n)
        return

    seen = 0
    for n in common:
        if args.filter and args.filter not in n:
            continue
        r = compare_one(n, bp, qp)
        seen += 1
        print(f"\n-- {r.get('name')}  shape={r.get('shape')}")
        if "error" in r:
            print(f"   error: {r['error']}")
            continue
        if "rmse_identity" in r:
            print(f"   identity:  rmse={r['rmse_identity']:.4g}  cos={r['cos_identity_mean']:.4f}")
        for k in ("lattice_bonsai", "lattice_base"):
            l = r.get(k)
            if l:
                print(f"   {k:14s}: groups={l.get('groups')} "
                      f"binary_frac={l.get('frac_binary_lattice'):.4f} "
                      f"max_distinct={l.get('max_distinct_per_group')} "
                      f"|μ|={l.get('mean_abs'):.4g}")
        rp = r.get("row_perm")
        if rp:
            print(f"   row_perm:  diag_cos={rp['identity_diag_cos']:.4f}  "
                  f"best_cos={rp['best_perm_cos']:.4f}  "
                  f"identity={rp['perm_is_identity']}")
        if seen >= 10:
            print("\n[... truncated; pass --filter to narrow]")
            break


if __name__ == "__main__":
    main()
