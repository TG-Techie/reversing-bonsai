# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Merge a sharded safetensors model into a single .safetensors file.

Reads each shard's header to find the data offsets, then copies the tensor
bytes verbatim into the merged output. Works for any dtype (including
BF16, which numpy can't open via the safetensors framework='numpy' path).

The Bonsai 4B / 8B unpacked releases are sharded; merging keeps the
existing analysis scripts unmodified and gives them complete coverage.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    return hdr_len, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_path")
    args = ap.parse_args()

    src = Path(args.src_dir)
    shard_files = sorted([p for p in src.glob("*.safetensors")
                          if p.is_file() and p.name != Path(args.out_path).name])
    if not shard_files:
        print("no safetensors shards in source dir", file=sys.stderr)
        sys.exit(2)

    # Build a unified key -> (shard_path, dtype, shape, src_off_a, src_off_b) map.
    # Each shard's data area starts after (8 + len(header)) bytes; src_off is
    # relative to data start.
    all_keys: dict[str, tuple] = {}
    shard_data_starts: dict[Path, int] = {}
    for sp in shard_files:
        hdr_len, hdr = read_header(sp)
        shard_data_starts[sp] = 8 + hdr_len
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            if k in all_keys:
                continue  # duplicates: take first
            a, b = meta["data_offsets"]
            all_keys[k] = (sp, meta["dtype"], meta["shape"], a, b)

    keys = sorted(all_keys.keys())
    print(f"merging {len(shard_files)} shards -> {len(keys)} unique tensors")

    # Build the merged header with tightly-packed offsets.
    merged_header: dict = {"__metadata__": {"merged_from": str(src)}}
    out_off = 0
    out_offsets: dict[str, int] = {}  # key -> dst start
    for k in keys:
        sp, dtype, shape, a, b = all_keys[k]
        nbytes = b - a
        merged_header[k] = {
            "dtype": dtype, "shape": shape,
            "data_offsets": [out_off, out_off + nbytes],
        }
        out_offsets[k] = out_off
        out_off += nbytes

    header_bytes = json.dumps(merged_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out_path}: header={header_len}, data={out_off}, total={8 + header_len + out_off}")

    BUF = 8 * 1024 * 1024  # 8 MiB
    # CRITICAL: write tensor bytes in the SAME ORDER as the merged header
    # claims. Header offsets were computed by iterating sorted(keys); we
    # MUST iterate sorted(keys) here too. An earlier per-shard-batched
    # version of this loop wrote bytes in shard order, which silently
    # mis-mapped tensors whose shapes happened to match (e.g. all q_proj
    # weights at a given size are the same shape, so a wrong layer's
    # q_proj loads without error and yields garbage cosines).
    src_handles: dict[Path, "BufferedReader"] = {}
    def src(sp: Path):
        if sp not in src_handles:
            src_handles[sp] = open(sp, "rb")
        return src_handles[sp]
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", header_len))
        out.write(header_bytes)
        for done, k in enumerate(keys, start=1):
            sp, _, _, a, b = all_keys[k]
            nbytes = b - a
            src_f = src(sp)
            src_f.seek(shard_data_starts[sp] + a)
            remaining = nbytes
            while remaining > 0:
                chunk = src_f.read(min(BUF, remaining))
                if not chunk:
                    print(f"unexpected EOF reading {k} from {sp}", file=sys.stderr)
                    sys.exit(2)
                out.write(chunk)
                remaining -= len(chunk)
            if done % 100 == 0 or done == len(keys):
                print(f"  [{done}/{len(keys)}] {k}  ({nbytes/1e6:.1f} MB)")
    for fh in src_handles.values():
        fh.close()
    print(f"[done] {out_path}  ({out_path.stat().st_size/1e9:.2f} GB)")

    # Verify: round-trip-load 3 random tensors and compare bytes to source.
    print("[verify] sampling 3 keys for round-trip byte-equality check...")
    import random
    rng = random.Random(0)
    sample_keys = rng.sample(keys, min(3, len(keys)))
    with open(out_path, "rb") as f_out:
        hdr_len = struct.unpack("<Q", f_out.read(8))[0]
        out_hdr = json.loads(f_out.read(hdr_len))
        out_data_start = 8 + hdr_len
        ok = True
        for k in sample_keys:
            sp, _, _, a, b = all_keys[k]
            with open(sp, "rb") as f_src:
                f_src.seek(shard_data_starts[sp] + a)
                src_bytes = f_src.read(b - a)
            oa, ob = out_hdr[k]["data_offsets"]
            f_out.seek(out_data_start + oa)
            out_bytes = f_out.read(ob - oa)
            if src_bytes == out_bytes:
                print(f"   OK   {k}  ({len(src_bytes)} bytes)")
            else:
                print(f"   FAIL {k}  (sizes {len(src_bytes)} vs {len(out_bytes)})")
                ok = False
        if not ok:
            sys.exit(3)


if __name__ == "__main__":
    main()
