# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Pure-Python Q1_0 GGUF block codec.

Block layout (matches llama.cpp ggml-common.h `block_q1_0`):
    struct {
        uint16_t d;                 // FP16 scale (mean of |x| in the group)
        uint8_t  qs[QK1_0 / 8];     // 16 bytes -> 128 sign bits
    };
    sizeof(block_q1_0) == 18,    QK1_0 == 128

Bit packing (matches llama.cpp quantize_row_q1_0_ref + dequantize_row_q1_0):
    bit at element index j -> qs[j/8] bit (j%8)
    bit value 1 -> +d
    bit value 0 -> -d

Therefore element ordering inside a block matches the row-major C ordering of
the underlying tensor: blocks are flat groups of 128 consecutive elements along
the last dim of the tensor (after row-major flattening).
"""

from __future__ import annotations

import numpy as np

QK1_0 = 128
BLOCK_BYTES = 2 + QK1_0 // 8  # = 18


def unpack_bytes_to_signs(qs: np.ndarray) -> np.ndarray:
    """qs shape (..., 16) uint8 -> shape (..., 128) of {0, 1} uint8."""
    qs = np.ascontiguousarray(qs, dtype=np.uint8)
    # unpackbits is bigendian within byte by default; we need bit 0 = LSB.
    bits = np.unpackbits(qs, axis=-1, bitorder="little")
    return bits


def parse_q1_0(buf: bytes | np.ndarray, n_elems: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Q1_0 raw byte buffer for a tensor of `n_elems` elements.

    Returns
    -------
    scales : (nblocks,) float32           per-group FP16 scale (cast to f32)
    signs  : (nblocks, 128) int8          {-1, +1}
    """
    if n_elems % QK1_0:
        raise ValueError(f"n_elems={n_elems} not divisible by QK1_0={QK1_0}")
    nb = n_elems // QK1_0
    raw = np.frombuffer(memoryview(buf).tobytes() if isinstance(buf, bytes) else buf,
                        dtype=np.uint8)
    if raw.size != nb * BLOCK_BYTES:
        raise ValueError(f"buf size {raw.size} != expected {nb * BLOCK_BYTES}")
    raw = raw.reshape(nb, BLOCK_BYTES)
    # FP16 scale is little-endian
    scales_fp16 = raw[:, :2].copy().view(np.float16).reshape(nb)
    scales = scales_fp16.astype(np.float32)
    qs = raw[:, 2:]  # (nb, 16)
    bits = unpack_bytes_to_signs(qs)  # (nb, 128) uint8 ∈ {0,1}
    signs = (bits.astype(np.int8) * 2) - 1  # {0,1} -> {-1, +1}
    return scales, signs


def dequantize_q1_0(buf: bytes | np.ndarray, n_elems: int) -> np.ndarray:
    """Return float32 tensor of n_elems values reconstructed from Q1_0 buffer."""
    scales, signs = parse_q1_0(buf, n_elems)
    out = signs.astype(np.float32) * scales[:, None]
    return out.reshape(n_elems)


def quantize_q1_0_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference quantizer matching llama.cpp `quantize_row_q1_0_ref`.

    Returns (scales_fp16_as_float32, signs_int8) per block.
    """
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if flat.size % QK1_0:
        raise ValueError(f"size {flat.size} not divisible by {QK1_0}")
    blocks = flat.reshape(-1, QK1_0)
    scales = np.mean(np.abs(blocks), axis=-1)  # NOTE: this is the d the ref uses
    scales_fp16 = scales.astype(np.float16).astype(np.float32)
    signs = np.where(blocks >= 0.0, np.int8(1), np.int8(-1))
    return scales_fp16, signs
