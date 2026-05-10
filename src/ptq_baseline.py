# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Post-Training Quantization (PTQ) baseline: sign-quantize Qwen3 base and
compare to Bonsai's actual weights.

If pure PTQ on Qwen3 produces something close to Bonsai, the case for QAT
weakens. If pure PTQ is materially *worse* than Bonsai (on the same
metrics we use elsewhere), we have direct empirical evidence that *some*
training-side intervention happened.

For each shared 2D weight tensor:
  1. Load Qwen3 base FP weights for that tensor.
  2. Group along the fast dim into 128-element blocks.
  3. Per-block: scale = mean(|w|); ptq_w = sign(w) * scale.
     This is exactly the ggml-quants.c reference Q1_0 quantizer.
  4. Compare ptq_w to Bonsai-unpacked using the SAME metrics as H2:
       - identity row cosine
       - sign agreement vs base
       - per-block scale s_g vs base mean(|w|)  (must be exact, by construction)
  5. Compare *Bonsai-unpacked* to Qwen3 base under the same metrics.
  6. Report the gap. If Bonsai's row cosine > PTQ's row cosine on the
     SAME tensor, Bonsai outperforms naive PTQ; the gap measures how much.

We expect: PTQ row cosine ≈ sqrt(2/π) ≈ 0.798 for Gaussian weights.
Bonsai's observed row cosine is 0.43-0.60. The hypothesis has been "Bonsai
is QAT'd, so its signs differ from sign(base), hence the lower cosine
*against* base; but its task accuracy is preserved." This script
quantifies the first half (the row-cosine gap).

Usage:
    uv run python src/ptq_baseline.py \
        models/q1/Bonsai-4B-Q1_0.gguf \
        models/unpacked/model-00001-of-00002.safetensors \
        models/base/model-00001-of-00003.safetensors \
        [--filter SUBSTR] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from q1_0 import QK1_0
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates, load_q1_tensor_from_gguf
from compare_unpacked_vs_qwen3 import load_tensor as load_st


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=-1) + 1e-12
    bn = np.linalg.norm(b, axis=-1) + 1e-12
    return np.einsum("ij,ij->i", a, b) / (an * bn)


def ptq_q1_0(base_arr: np.ndarray, group: int = QK1_0) -> np.ndarray:
    """Reference PTQ Q1_0 of the base array along its FAST dim."""
    flat = base_arr.reshape(-1).astype(np.float32)
    if flat.size % group:
        raise ValueError(f"size {flat.size} not multiple of {group}")
    blocks = flat.reshape(-1, group)
    scales = np.mean(np.abs(blocks), axis=-1)  # ggml-quants.c rule
    # Cast to FP16 and back, mirroring what GGUF storage would do
    scales = scales.astype(np.float16).astype(np.float32)
    signs = np.where(blocks >= 0.0, np.float32(1.0), np.float32(-1.0))
    out = signs * scales[:, None]
    return out.reshape(base_arr.shape).astype(np.float32)


def compare_one(name: str, q1: GGUFReader, unp: Path, base: Path) -> dict:
    out: dict = {"gguf_name": name}
    try:
        deq, hf_shape, _, _ = load_q1_tensor_from_gguf(q1, name)
    except (KeyError, TypeError) as e:
        return {"gguf_name": name, "error": str(e)}
    cand = gguf_to_hf_candidates(name)
    out["hf_candidates"] = cand
    bonsai = None
    base_arr = None
    found = None
    for c in cand:
        bonsai = load_st(unp, c)
        if bonsai is not None:
            found = c
            base_arr = load_st(base, c)
            break
    if bonsai is None or base_arr is None:
        return {"gguf_name": name, "error": "tensor missing in safetensors"}
    if bonsai.shape != base_arr.shape:
        return {"gguf_name": name, "error": f"shape mismatch {bonsai.shape} vs {base_arr.shape}"}
    out["hf_name"] = found
    out["shape"] = list(bonsai.shape)

    # Bonsai vs base
    bonsai = bonsai.astype(np.float32)
    base = base_arr.astype(np.float32)
    cos_bonsai = float(cos_per_row(bonsai, base).mean())

    # PTQ-of-base vs base
    ptq = ptq_q1_0(base)
    cos_ptq_vs_base = float(cos_per_row(ptq, base).mean())

    # PTQ vs Bonsai (do the two quantizers agree?)
    cos_ptq_vs_bonsai = float(cos_per_row(ptq, bonsai).mean())

    # Sign agreements
    sign_bonsai_vs_base = float(np.mean(np.sign(bonsai) == np.sign(base)))
    sign_ptq_vs_base = float(np.mean(np.sign(ptq) == np.sign(base)))
    sign_ptq_vs_bonsai = float(np.mean(np.sign(ptq) == np.sign(bonsai)))

    out.update({
        "cos_bonsai_vs_base":     cos_bonsai,
        "cos_ptq_vs_base":        cos_ptq_vs_base,
        "cos_ptq_vs_bonsai":      cos_ptq_vs_bonsai,
        "sign_bonsai_vs_base":    sign_bonsai_vs_base,
        "sign_ptq_vs_base":       sign_ptq_vs_base,
        "sign_ptq_vs_bonsai":     sign_ptq_vs_bonsai,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("unpacked")
    ap.add_argument("base")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    q1 = GGUFReader(args.gguf, "r")
    unp = Path(args.unpacked)
    base = Path(args.base)
    q1_tensors = [t for t in q1.tensors if t.tensor_type == GGMLQuantizationType.Q1_0]
    print(f"== {len(q1_tensors)} Q1_0 tensors")

    rows = []
    for t in q1_tensors:
        if args.filter and args.filter not in t.name:
            continue
        r = compare_one(t.name, q1, unp, base)
        if "error" in r:
            print(f"\n-- {r['gguf_name']}\n   error: {r['error']}")
            continue
        rows.append(r)
        print(
            f"\n-- {r['gguf_name']} -> {r['hf_name']}  shape={r['shape']}\n"
            f"   cos(Bonsai, base)             = {r['cos_bonsai_vs_base']:.4f}\n"
            f"   cos(PTQ-of-base, base)        = {r['cos_ptq_vs_base']:.4f}    <- naive Q1_0 of Qwen3\n"
            f"   cos(PTQ-of-base, Bonsai)      = {r['cos_ptq_vs_bonsai']:.4f}\n"
            f"   sign(Bonsai)==sign(base)      = {r['sign_bonsai_vs_base']*100:.2f}%\n"
            f"   sign(PTQ)==sign(base)         = {r['sign_ptq_vs_base']*100:.2f}%  <- 100% by construction\n"
            f"   sign(PTQ)==sign(Bonsai)       = {r['sign_ptq_vs_bonsai']*100:.2f}%"
        )
        if args.limit and len(rows) >= args.limit:
            break

    if rows:
        print("\n== AGGREGATE ==")
        for k, label in [
            ("cos_bonsai_vs_base",     "cos(Bonsai vs base)"),
            ("cos_ptq_vs_base",        "cos(PTQ-of-base vs base)"),
            ("cos_ptq_vs_bonsai",      "cos(PTQ-of-base vs Bonsai)"),
            ("sign_bonsai_vs_base",    "sign agree(Bonsai, base)"),
            ("sign_ptq_vs_base",       "sign agree(PTQ, base)"),
            ("sign_ptq_vs_bonsai",     "sign agree(PTQ, Bonsai)"),
        ]:
            v = np.array([r[k] for r in rows], dtype=np.float64)
            print(f"  {label:38s}  mean={v.mean():.4f}  min={v.min():.4f}  max={v.max():.4f}")
        gap = np.array([r["cos_ptq_vs_base"] - r["cos_bonsai_vs_base"] for r in rows])
        print(f"\n  PTQ_cos − Bonsai_cos  (per tensor)")
        print(f"     mean={gap.mean():.4f}  min={gap.min():.4f}  max={gap.max():.4f}")
        print()
        print("  Reading guide:")
        print("    cos(PTQ vs base)    ~  0.80   sign-quant of Gaussian w tracks at ≈ sqrt(2/π).")
        print("    cos(Bonsai vs base) <  0.80   means Bonsai's signs differ from sign(base).")
        print("    The gap (PTQ − Bonsai) measures how much retraining moved the model away from base.")
        print("    cos(PTQ vs Bonsai)            measures how close naive Q1_0(base) is to Bonsai itself.")


if __name__ == "__main__":
    main()
