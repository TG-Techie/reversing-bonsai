# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Dequantize a Bonsai Q1_0 GGUF into a HuggingFace-style safetensors file.

Why: H1 (verified at 1.7B and 4B) shows that Bonsai-unpacked is byte-equal
to dequant(Q1_0) modulo FP16 storage precision. So for sizes where the
unpacked safetensors isn't conveniently available (8B, where the workflow
deliberately skips it to fit the runner timeout), we can derive a
unpacked-equivalent locally from the GGUF.

The output is a single-file safetensors with HF tensor names
(e.g. `model.layers.0.self_attn.q_proj.weight`) and FP16 dtype, which is
exactly the format `compare_unpacked_vs_qwen3.py`, `compare_magnitudes.py`,
`test_column_permutation.py`, and `sign_disagreement.py` consume.

Usage:
    uv run python src/dequantize_gguf_to_safetensors.py \\
        models/bonsai/8B/gguf/Bonsai-8B-Q1_0.gguf \\
        models/bonsai/8B/unpacked/model.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import parse_q1_0
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out", help="Output safetensors path")
    args = ap.parse_args()

    reader = GGUFReader(args.gguf, "r")
    out = {}
    n_q1 = 0
    n_other = 0
    for t in reader.tensors:
        # Map to HF name. We pass through F32 / F16 tensors verbatim under
        # their HF-renamed key so the file is consumable as a drop-in for
        # the unpacked safetensors.
        hf_names = gguf_to_hf_candidates(t.name)
        if not hf_names:
            continue
        hf = hf_names[0]
        gguf_shape = list(t.shape)
        hf_shape = list(reversed(gguf_shape))  # GGUF is fastest-dim-first
        if t.tensor_type == GGMLQuantizationType.Q1_0:
            raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
            scales, signs = parse_q1_0(raw, int(np.prod(gguf_shape)))
            arr = (signs.astype(np.float32) * scales[:, None]).reshape(-1)
            out[hf] = arr.reshape(hf_shape).astype(np.float16)
            n_q1 += 1
        else:
            arr = np.asarray(t.data)
            # Some tensor types come back already in target order
            if arr.size == int(np.prod(gguf_shape)):
                arr = arr.reshape(hf_shape)
            out[hf] = arr.astype(np.float16) if arr.dtype != np.float32 else arr
            n_other += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    print(f"Wrote {len(out)} tensors to {args.out}")
    print(f"  Q1_0 dequantized: {n_q1}")
    print(f"  passthrough:      {n_other}")


if __name__ == "__main__":
    main()
