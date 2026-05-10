# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Merge a sharded safetensors model (model-NNNNN-of-MMMMM.safetensors,
guided by model.safetensors.index.json) into a single .safetensors file.

The Bonsai-1.7B-unpacked release ships as a single file but the 4B and 8B
unpacked releases (and the Qwen3 base) are sharded. Our analysis scripts
take one safetensors file, so a one-shot merge keeps them unmodified.

This streams tensor-by-tensor; peak memory is ≈ the largest tensor.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="Directory containing model-*-of-*.safetensors + model.safetensors.index.json")
    ap.add_argument("out_path", help="Destination .safetensors path")
    args = ap.parse_args()

    src = Path(args.src_dir)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.is_file():
        # Fall back to listing all safetensors files; collect their key sets
        shards = sorted(src.glob("*.safetensors"))
        if not shards:
            print("no safetensors files in source dir", file=sys.stderr)
            sys.exit(2)
        weight_map = {}
        for sf in shards:
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    weight_map[k] = sf.name
    else:
        weight_map = json.loads(idx_path.read_text()).get("weight_map", {})

    if not weight_map:
        print("empty weight_map", file=sys.stderr)
        sys.exit(2)

    # Open one handle per shard (lazy, mmap-backed via safetensors)
    shards: dict[str, "safe_open"] = {}
    def get(shard_name: str):
        if shard_name not in shards:
            shards[shard_name] = safe_open(str(src / shard_name), framework="numpy")
            shards[shard_name].__enter__()
        return shards[shard_name]

    # Build header with offsets into the merged data area
    # safetensors layout: [u64 header_len LE] [JSON header bytes] [data bytes]
    keys = sorted(weight_map.keys())
    header = {"__metadata__": {"merged_from": str(src)}}
    offset = 0
    tensor_metas = {}  # name -> (shard, dtype, shape, off_a, off_b)
    for k in keys:
        shard = weight_map[k]
        f = get(shard)
        # Read meta from the shard's own header
        meta = f.metadata()
        # safetensors safe_open has `get_slice` that gives shape & dtype; use it
        sl = f.get_slice(k)
        shape = list(sl.get_shape())
        dtype = sl.get_dtype()
        # Compute byte size for this dtype
        elem_bits = {
            "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
            "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8,
            "BOOL": 8,
        }[dtype]
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = (nelem * elem_bits) // 8
        header[k] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        tensor_metas[k] = (shard, dtype, shape, offset, offset + nbytes, nbytes)
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires header to be 8-byte aligned-padded? Actually it's
    # whatever the producer wrote; HF pads to 8 bytes for performance. We'll do that.
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"writing {args.out_path}: {len(keys)} tensors, header={header_len}, data={offset}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", header_len))
        out.write(header_bytes)
        # Now stream each tensor's bytes from its source shard
        for i, k in enumerate(keys):
            shard, dtype, shape, off_a, off_b, nbytes = tensor_metas[k]
            arr = get(shard).get_tensor(k)
            # arr is a numpy view in the source dtype
            buf = arr.tobytes()
            if len(buf) != nbytes:
                raise ValueError(f"size mismatch for {k}: arr={len(buf)} expected={nbytes}")
            out.write(buf)
            if (i + 1) % 50 == 0 or (i + 1) == len(keys):
                print(f"  [{i+1}/{len(keys)}] {k}  {nbytes/1e6:.1f}MB")

    for f in shards.values():
        f.__exit__(None, None, None)
    print(f"[done] {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
