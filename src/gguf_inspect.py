"""Print GGUF metadata + tensor inventory.

Usage: uv run python src/gguf_inspect.py <path.gguf> [--values KEY ...] [--tensors]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gguf import GGUFReader, GGMLQuantizationType


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--values", nargs="*", default=[],
                    help="Metadata keys to print full value of (default: just type+shape)")
    ap.add_argument("--tensors", action="store_true",
                    help="Print full tensor inventory")
    ap.add_argument("--filter", default=None,
                    help="Substring filter for tensor names")
    args = ap.parse_args(argv)

    r = GGUFReader(args.path, "r")

    print(f"== {args.path} ==")
    # GGUFReader doesn't expose a `version` attribute in all gguf releases;
    # try the typical names but don't fail if missing.
    ver = getattr(r, "version", None) or getattr(r, "gguf_version", None) or "?"
    print(f"  GGUF version: {ver}")
    print(f"  alignment:    {getattr(r, 'alignment', '?')}")
    print(f"  KV pairs:     {len(r.fields)}")
    print(f"  tensors:      {len(r.tensors)}")

    print("\n== METADATA ==")
    for k, field in r.fields.items():
        try:
            t = field.types[0].name
        except Exception:
            t = "?"
        if k in args.values:
            try:
                v = r.get_field(k).contents()
            except Exception:
                v = field.parts
            sval = repr(v)
            if len(sval) > 400:
                sval = sval[:400] + "..."
            print(f"  {k} :: {t} = {sval}")
        else:
            n = len(field.data) if hasattr(field, "data") else "?"
            print(f"  {k} :: {t} (n={n})")

    if args.tensors:
        print("\n== TENSORS ==")
        # quant types breakdown
        bytype = {}
        total_bytes = 0
        for t in r.tensors:
            bytype.setdefault(t.tensor_type.name, [0, 0])
            bytype[t.tensor_type.name][0] += 1
            bytype[t.tensor_type.name][1] += t.n_bytes
            total_bytes += t.n_bytes
        print("  Quant breakdown:")
        for k, (n, b) in sorted(bytype.items(), key=lambda x: -x[1][1]):
            print(f"    {k:12s}  count={n:4d}  bytes={fmt_size(b):>12s}")
        print(f"  Total tensor bytes: {fmt_size(total_bytes)}")

        print("\n  Per-tensor (filtered):")
        flt = args.filter
        for t in r.tensors:
            if flt and flt not in t.name:
                continue
            print(f"    {t.tensor_type.name:10s}  shape={list(t.shape)!s:30s}  "
                  f"bytes={fmt_size(t.n_bytes):>12s}  {t.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
