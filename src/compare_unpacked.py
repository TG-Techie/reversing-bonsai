# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Compare three artifacts:

  1. Bonsai-{size}-Q1_0.gguf      (1-bit packed)
  2. Bonsai-{size} F16 / unpacked (full-precision dump of the same Bonsai weights)
  3. Qwen3-{size} base            (the model Bonsai was made FROM)

Hypothesis trio:
  H1 (lossless dequant):
       dequantize(Q1_0).reshape == unpacked-FP-tensor exactly  (≤1 ULP FP16)
       => Bonsai's "unpacked" is just FP16 holding the binary-quantized values.

  H2 (no permutation):
       Bonsai unpacked tensors share the SAME shape + ordering as Qwen3 base
       => can be directly diffed element-wise.

  H3 (sortedness within group):
       within each 128-element block, signs are arranged in some order.
       Random => transitions ~ 64 (mean of binomial-of-127). Sorted => 0 or 1.
       If Bonsai used row/col permutations to cluster sign within group, the
       Q1_0 signs would come pre-sorted while the original FP weights had
       the same signs but interleaved.

We test:
  A) For each Q1_0 tensor in Bonsai GGUF, parse it.
     If a "unpacked" GGUF (all F16) is provided, look up the same tensor name
     and compare element-wise.
  B) If Qwen3 base is provided (safetensors / GGUF), look up matching tensor
     name, then for each row compute:
       - sign agreement rate against Bonsai signs
       - cosine similarity
       - per-row scale fit (||q3|| / sqrt(N))
       - whether |row of Qwen3| has 128-block-piecewise-constant magnitude

Usage:
    uv run python src/compare_unpacked.py --bonsai-q1 <file> [--bonsai-fp <file>] [--qwen <file>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import QK1_0, parse_q1_0, dequantize_q1_0


def load_gguf_tensor(reader: GGUFReader, name: str) -> tuple[np.ndarray, str, list[int]]:
    for t in reader.tensors:
        if t.name == name:
            shape = list(t.shape)
            tt = t.tensor_type
            if tt == GGMLQuantizationType.F32:
                arr = np.array(t.data).view(np.float32).reshape(shape[::-1]).astype(np.float32)
            elif tt == GGMLQuantizationType.F16:
                arr = np.array(t.data).view(np.float16).reshape(shape[::-1]).astype(np.float32)
            elif tt == GGMLQuantizationType.BF16:
                # bf16: stored as uint16 with the upper half of f32 bits
                u16 = np.array(t.data).view(np.uint16).reshape(shape[::-1])
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32)
            elif tt == GGMLQuantizationType.Q1_0:
                # use our pure-py dequantizer
                n_elems = int(np.prod(shape))
                raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
                arr = dequantize_q1_0(raw, n_elems).reshape(shape[::-1])
            else:
                raise NotImplementedError(f"loader for {tt.name} not implemented")
            return arr, tt.name, shape
    raise KeyError(name)


def row_sign_diagnostics(row: np.ndarray, k: int = 128) -> dict:
    """`row` 1D float, length divisible by k. Per-block stats."""
    n = row.size
    if n % k:
        raise ValueError
    blocks = row.reshape(-1, k)
    pos = np.sum(blocks > 0, axis=1)
    neg = np.sum(blocks < 0, axis=1)
    zero = np.sum(blocks == 0, axis=1)
    signs = np.sign(blocks)
    transitions = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    abs_block_max = np.max(np.abs(blocks), axis=1)
    abs_block_min = np.min(np.abs(blocks), axis=1)
    abs_uniform = np.isclose(abs_block_max, abs_block_min, rtol=0, atol=1e-6)
    return {
        "transitions_mean": float(transitions.mean()),
        "transitions_min": int(transitions.min()),
        "transitions_max": int(transitions.max()),
        "frac_<=1trans": float((transitions <= 1).mean()),
        "frac_uniform_abs": float(abs_uniform.mean()),
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "zero_mean": float(zero.mean()),
    }


def compare_tensors_pair(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        print(f"   shape mismatch: {name_a}={a.shape} vs {name_b}={b.shape}")
        return
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    rel = diff / np.where(np.abs(b) > 1e-12, np.abs(b), 1.0)
    same_sign = float(np.mean(np.sign(a) == np.sign(b)))
    print(f"   ||a-b||_inf = {np.max(np.abs(diff)):.4g}  "
          f"||a-b||_2/||b||_2 = {np.linalg.norm(diff)/(np.linalg.norm(b)+1e-12):.4g}  "
          f"sign agree = {same_sign*100:.4f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonsai-q1", required=True)
    ap.add_argument("--bonsai-fp", default=None,
                    help="GGUF or safetensors of unpacked Bonsai (FP)")
    ap.add_argument("--qwen", default=None,
                    help="GGUF or safetensors of base Qwen3")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    rq = GGUFReader(args.bonsai_q1, "r")
    rfp = GGUFReader(args.bonsai_fp, "r") if args.bonsai_fp else None
    rqwen = GGUFReader(args.qwen, "r") if args.qwen else None

    q1_tensors = [t for t in rq.tensors if t.tensor_type == GGMLQuantizationType.Q1_0]
    print(f"Bonsai Q1: {len(q1_tensors)} Q1_0 tensors")

    for i, t in enumerate(q1_tensors[: args.limit]):
        print(f"\n[{i}] {t.name}  shape={list(t.shape)}")
        arr_q1, _, shape = load_gguf_tensor(rq, t.name)
        # Per-row diagnostic on unpacked Q1
        flat = arr_q1.reshape(-1)
        n = flat.size
        if n % QK1_0:
            print("   skip: not divisible by 128")
            continue
        # row diagnostics on each row of the 2D tensor (last dim is fast)
        if arr_q1.ndim == 2:
            for r_idx in range(min(2, arr_q1.shape[0])):
                d = row_sign_diagnostics(arr_q1[r_idx], k=QK1_0)
                print(f"   Q1 row {r_idx}: trans μ={d['transitions_mean']:.2f} "
                      f"max={d['transitions_max']} <=1={d['frac_<=1trans']*100:.2f}%  "
                      f"unif|·|={d['frac_uniform_abs']*100:.1f}%")

        if rfp is not None:
            try:
                arr_fp, kind, shp = load_gguf_tensor(rfp, t.name)
                print(f"   matched in FP file (type={kind}, shape={shp})")
                compare_tensors_pair(arr_q1, arr_fp, "q1", "fp")
            except KeyError:
                print(f"   not found in FP: {t.name}")
            except Exception as e:
                print(f"   FP compare error: {e}")

        if rqwen is not None:
            try:
                arr_qw, kind, shp = load_gguf_tensor(rqwen, t.name)
                print(f"   matched in Qwen3 (type={kind}, shape={shp})")
                # baseline diff
                if arr_qw.shape == arr_q1.shape:
                    a = arr_q1.astype(np.float64)
                    b = arr_qw.astype(np.float64)
                    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    sign_eq = float(np.mean(np.sign(a) == np.sign(b)))
                    print(f"   vs Qwen: cos={cos:.4f}  sign-agree={sign_eq*100:.2f}%")
                    if arr_qw.ndim == 2:
                        for r_idx in range(min(2, arr_qw.shape[0])):
                            d = row_sign_diagnostics(arr_qw[r_idx], k=QK1_0)
                            print(f"   Qwen row {r_idx}: trans μ={d['transitions_mean']:.2f} "
                                  f"unif|·|={d['frac_uniform_abs']*100:.1f}%")
            except KeyError:
                print(f"   not found in Qwen: {t.name}")
            except Exception as e:
                print(f"   Qwen compare error: {e}")


if __name__ == "__main__":
    main()
