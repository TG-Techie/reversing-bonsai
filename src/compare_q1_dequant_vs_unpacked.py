"""Compare dequantize(Bonsai-Q1_0 GGUF) vs Bonsai-unpacked safetensors.

This is the missing bridge between the existing tools:
  * `compare_unpacked.py` requires the unpacked file in GGUF form (skipped in
    the workflow because Bonsai-unpacked ships as safetensors).
  * `compare_unpacked_vs_qwen3.py` compares two FP files but does not read Q1_0.

What this script answers (Hypothesis A from FINDINGS.md):
   "If `Bonsai-unpacked` and `dequant(Bonsai-Q1_0)` agree to fp16 precision
    element-wise, the unpacked file is the identical binary lattice in FP16
    storage."

Per matched tensor it reports:
  - shape match (after GGUF axis reversal — GGUF stores fastest-dim-first)
  - max-abs diff between dequant(Q1_0) and the unpacked f16 tensor
  - whether all groups of 128 weights along the fast dim live on a single
    magnitude in the unpacked tensor (a binary lattice signature)
  - whether the per-group scale d in Q1_0 == mean(|w|) of the corresponding
    unpacked group (i.e. ggml-quants.c rule)

GGUF naming uses `blk.<i>.attn_q.weight` etc; safetensors uses HF's
`model.layers.<i>.self_attn.q_proj.weight`. We translate both directions.

Usage:
    uv run python src/compare_q1_dequant_vs_unpacked.py \
        models/gguf/Bonsai-1.7B-Q1_0.gguf \
        models/unpacked/<unpacked.safetensors> [--filter <substr>] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import QK1_0, dequantize_q1_0, parse_q1_0


# Map GGUF -> HF (safetensors) tensor names. Returns a list of candidates.
def gguf_to_hf_candidates(name: str) -> list[str]:
    # Strip the trailing ".weight" so we map the body, then reattach.
    if not name.endswith(".weight") and not name.endswith(".bias"):
        return [name]
    suffix = ".weight" if name.endswith(".weight") else ".bias"
    body = name[: -len(suffix)]
    # token_embd / output / blk.<i>.<sub>
    if body == "token_embd":
        return [f"model.embed_tokens{suffix}"]
    if body == "output":
        return [f"lm_head{suffix}"]
    m = re.match(r"^blk\.(\d+)\.(.+)$", body)
    if not m:
        return [name]
    idx, sub = m.group(1), m.group(2)
    sub_map = {
        "attn_q":     "self_attn.q_proj",
        "attn_k":     "self_attn.k_proj",
        "attn_v":     "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_gate":   "mlp.gate_proj",
        "ffn_up":     "mlp.up_proj",
        "ffn_down":   "mlp.down_proj",
        "attn_norm":  "input_layernorm",
        "ffn_norm":   "post_attention_layernorm",
        # Qwen3-specific qk-norms (head-wise RMSNorm on q/k):
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
    }
    hf_sub = sub_map.get(sub)
    if hf_sub is None:
        return [f"model.layers.{idx}.{sub}{suffix}"]
    return [f"model.layers.{idx}.{hf_sub}{suffix}"]


def load_q1_tensor_from_gguf(reader: GGUFReader, name: str) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Return (dequant_arr, shape_hf, scales, signs).

    dequant_arr is (out, in) float32 — i.e. HF orientation, since GGUF stores
    fastest-dim-first and reshape(shape[::-1]) restores HF order.
    """
    for t in reader.tensors:
        if t.name != name:
            continue
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            raise TypeError(f"{name} is {t.tensor_type.name}, not Q1_0")
        gguf_shape = list(t.shape)               # fastest-dim first
        hf_shape = list(reversed(gguf_shape))    # safetensors orientation
        n_elems = int(np.prod(gguf_shape))
        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        scales, signs = parse_q1_0(raw, n_elems)
        flat = (signs.astype(np.float32) * scales[:, None]).reshape(-1)
        arr = flat.reshape(hf_shape)
        return arr, hf_shape, scales, signs
    raise KeyError(name)


def lattice_per_group(unpacked_row: np.ndarray, group: int = 128) -> dict:
    n = unpacked_row.size
    if n % group:
        return {"error": f"size {n} not multiple of {group}"}
    g = unpacked_row.reshape(-1, group)
    abs_g = np.abs(g.astype(np.float32))
    rounded = np.round(abs_g * 1e6) / 1e6
    distinct = np.array([np.unique(r).size for r in rounded])
    return {
        "groups": int(g.shape[0]),
        "frac_binary_lattice": float((distinct == 1).mean()),
        "max_distinct": int(distinct.max()),
    }


def compare_one(name: str, gguf: GGUFReader, unpacked_path: Path, limit_rows: int = 64) -> dict:
    out: dict = {"gguf_name": name}
    try:
        deq, hf_shape, scales, signs = load_q1_tensor_from_gguf(gguf, name)
    except (KeyError, TypeError) as e:
        return {"gguf_name": name, "error": str(e)}

    # Map to HF name
    hf_candidates = gguf_to_hf_candidates(name)
    out["hf_candidates"] = hf_candidates

    # Try to load the matching tensor from safetensors index
    from safetensors import safe_open
    found_name = None
    arr_un = None
    # If the path is a single safetensors, load directly. If it's a directory or
    # a .json index, scan all shards.
    if unpacked_path.is_file() and unpacked_path.suffix == ".safetensors":
        with safe_open(str(unpacked_path), framework="numpy") as f:
            keys = set(f.keys())
            for cand in hf_candidates:
                if cand in keys:
                    found_name = cand
                    arr_un = f.get_tensor(cand).astype(np.float32)
                    break
    else:
        # Directory: scan every safetensors file
        sf_files = sorted(Path(unpacked_path).glob("*.safetensors")) if unpacked_path.is_dir() \
                   else sorted(Path(unpacked_path).parent.glob("*.safetensors"))
        for sf in sf_files:
            with safe_open(str(sf), framework="numpy") as f:
                keys = set(f.keys())
                for cand in hf_candidates:
                    if cand in keys:
                        found_name = cand
                        arr_un = f.get_tensor(cand).astype(np.float32)
                        break
            if arr_un is not None:
                break

    if arr_un is None:
        out["error"] = f"unpacked tensor not found (candidates: {hf_candidates})"
        return out
    out["hf_name"] = found_name
    out["dequant_shape"] = list(deq.shape)
    out["unpacked_shape"] = list(arr_un.shape)

    # Shape compare
    if tuple(deq.shape) != tuple(arr_un.shape):
        out["error"] = "shape mismatch"
        return out

    # Element-wise diff (max abs)
    diff = deq - arr_un
    out["max_abs_diff"] = float(np.max(np.abs(diff)))
    out["mean_abs_diff"] = float(np.mean(np.abs(diff)))
    # rmse relative to mean magnitude of unpacked
    base_mag = float(np.mean(np.abs(arr_un)) + 1e-12)
    out["rel_rmse"] = float(np.sqrt(np.mean(diff**2)) / base_mag)
    # Sign agreement (where unpacked != 0)
    nz = np.abs(arr_un) > 0
    out["sign_agree"] = float(np.mean(np.sign(deq[nz]) == np.sign(arr_un[nz])))

    # Lattice property of unpacked: group of 128 along fast dim, one |w|?
    # We pass the flat row-major view (last dim is fast in HF).
    out["lattice_unpacked"] = lattice_per_group(arr_un.reshape(-1), QK1_0)

    # Scale fidelity: does the FP16 d in Q1_0 == mean(|x|) of unpacked group?
    flat_un = arr_un.reshape(-1)
    if flat_un.size % QK1_0 == 0:
        groups_un = flat_un.reshape(-1, QK1_0)
        scales_from_unpacked = np.mean(np.abs(groups_un.astype(np.float32)), axis=-1)
        # scales is shape (nblocks,) f32 from FP16
        s_diff = scales_from_unpacked - scales
        out["scale_max_abs_diff"] = float(np.max(np.abs(s_diff)))
        out["scale_mean_abs_diff"] = float(np.mean(np.abs(s_diff)))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="Path to Bonsai-*-Q1_0.gguf")
    ap.add_argument("unpacked", help="Path to a *.safetensors file or its parent directory")
    ap.add_argument("--filter", default=None, help="Only compare GGUF tensor names containing this substring")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tensors compared (0 = all)")
    args = ap.parse_args()

    reader = GGUFReader(args.gguf, "r")
    unpacked = Path(args.unpacked)

    q1_tensors = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.Q1_0]
    print(f"== Q1_0 tensors in {args.gguf}: {len(q1_tensors)}")

    seen = 0
    worst_max = 0.0
    worst_name = None
    bin_lattice_count = 0
    bin_lattice_total = 0
    scale_diffs = []
    for t in q1_tensors:
        if args.filter and args.filter not in t.name:
            continue
        r = compare_one(t.name, reader, unpacked)
        seen += 1
        if "error" in r:
            print(f"\n-- {r['gguf_name']}\n   error: {r['error']}")
            continue
        line = (
            f"\n-- {r['gguf_name']} -> {r['hf_name']}  shape={r['dequant_shape']}\n"
            f"   max|deq - unpacked|         = {r['max_abs_diff']:.4g}\n"
            f"   mean|deq - unpacked|        = {r['mean_abs_diff']:.4g}\n"
            f"   rel_rmse vs mean(|unpacked|)= {r['rel_rmse']:.4g}\n"
            f"   sign agreement (nonzero)    = {r['sign_agree']*100:.4f}%\n"
            f"   unpacked lattice            = "
            f"groups={r['lattice_unpacked'].get('groups')} "
            f"frac_binary={r['lattice_unpacked'].get('frac_binary_lattice'):.4f} "
            f"max_distinct={r['lattice_unpacked'].get('max_distinct')}"
        )
        if "scale_max_abs_diff" in r:
            line += (
                f"\n   scale max|d_q1 - mean(|x|)|  = {r['scale_max_abs_diff']:.4g}"
                f"\n   scale mean|·|                = {r['scale_mean_abs_diff']:.4g}"
            )
            scale_diffs.append(r["scale_mean_abs_diff"])
        print(line)

        # Aggregate
        if r["max_abs_diff"] > worst_max:
            worst_max = r["max_abs_diff"]
            worst_name = r["gguf_name"]
        l = r.get("lattice_unpacked", {})
        if isinstance(l, dict) and "groups" in l:
            bin_lattice_count += int(round(l["frac_binary_lattice"] * l["groups"]))
            bin_lattice_total += l["groups"]
        if args.limit and seen >= args.limit:
            break

    print("\n== SUMMARY ==")
    print(f"  tensors compared:           {seen}")
    print(f"  worst max|deq - unpacked|:  {worst_max:.4g}  ({worst_name})")
    if bin_lattice_total:
        print(f"  global binary-lattice frac: {bin_lattice_count / bin_lattice_total:.6f}"
              f"  ({bin_lattice_count}/{bin_lattice_total} groups)")
    if scale_diffs:
        print(f"  mean of per-tensor scale-mean-diffs: {np.mean(scale_diffs):.4g}")
    print()
    print("Verdict guidance:")
    print("  - If max|deq - unpacked| <= ~5e-4 across all tensors AND lattice frac == 1.0:")
    print("    => Hypothesis A holds: unpacked is FP16 storage of dequant(Q1_0).")
    print("  - If max|deq - unpacked| is small but lattice frac < 1.0:")
    print("    => unpacked has FP residuals (likely from QAT pre-quantization training).")


if __name__ == "__main__":
    main()
