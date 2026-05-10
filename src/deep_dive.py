# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Rigorous + creative reverse-engineering follow-ups on Bonsai vs Qwen3.

Focused on questions that go beyond the existing H1-H4 / magnitude
analyses. Each section is independent and prints its own block; aggregate
runtime is a few minutes per (size).

Sections:
    A. Norm-tensor equality. Are the 1D F32 RMSNorm tensors
       (attn_norm / ffn_norm / model.norm / per-head q_norm / k_norm)
       byte-identical between Bonsai-unpacked and Qwen3-base, or were
       they re-trained? If identical, PrismML preserved the
       normalization layers verbatim — a strong hint that QAT touched
       only the matrix-heavy weights.
    B. Embedding vocab diff. Which 267 vocab entries does Bonsai-1.7B
       drop relative to Qwen3-1.7B? (Inferred from shape mismatch we
       saw under H2.) Spot-check the dropped tokens to see if they're
       all reserved/special.
    C. LM-head tying. Bonsai-unpacked has fewer top-level keys than
       Qwen3 base; did PrismML weight-tie LM head to embed_tokens, or
       just drop the LM head entirely?
    D. Cosine breakdown by projection type. Aggregate H2 cosine
       (Bonsai-unpacked vs Qwen3-base) by tensor role (q / k / v / o /
       gate / up / down). Pattern hints at where QAT did most work.
    E. Per-head magnitude profile. Split attn_v / attn_q / attn_k along
       the head dimension and compare per-head mean(|w|) Bonsai vs
       Qwen3. Are there heads where Bonsai is dramatically louder /
       quieter than the base?
    F. Cosine outliers. For each weight matrix, find the top-K rows
       where Bonsai-vs-base cosine is most off. Are they clustered at
       a specific layer / projection / head?
    G. Scale-distribution shape. The ggml-quants.c reference Q1_0 sets
       scale = mean(|x|). For Gaussian x, that's σ * sqrt(2/π). For a
       generic positive distribution, mean(|x|) / std(x) is a shape
       statistic. We compute it per Bonsai block and compare to the
       Gaussian reference and to Qwen3's per-group ratio.

Usage:
    uv run python src/deep_dive.py \
        models/q1/Bonsai-4B-Q1_0.gguf \
        models/unpacked/model.safetensors \
        models/base/model.safetensors
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from q1_0 import QK1_0, parse_q1_0
from compare_unpacked_vs_qwen3 import load_tensor, list_tensor_names


# -----------------------------------------------------------------------------
# A. Norm-tensor equality
# -----------------------------------------------------------------------------
NORM_PATTERNS = [
    r"^model\.norm\.weight$",
    r"^model\.layers\.\d+\.input_layernorm\.weight$",
    r"^model\.layers\.\d+\.post_attention_layernorm\.weight$",
    r"^model\.layers\.\d+\.self_attn\.q_norm\.weight$",
    r"^model\.layers\.\d+\.self_attn\.k_norm\.weight$",
]


def section_A_norm_equality(unp: Path, base: Path) -> dict:
    print("\n== A. norm-tensor equality (Bonsai-unpacked vs Qwen3-base) ==")
    bk = set(list_tensor_names(unp))
    qk = set(list_tensor_names(base))
    matches = sorted(k for k in bk & qk if any(re.match(p, k) for p in NORM_PATTERNS))
    by_kind = defaultdict(lambda: {"n": 0, "exact": 0, "max_diff": 0.0})
    for k in matches:
        bw = load_tensor(unp, k)
        qw = load_tensor(base, k)
        if bw is None or qw is None:
            continue
        if bw.shape != qw.shape:
            kind = k.split(".")[-2]  # e.g. "input_layernorm" or "q_norm"
            by_kind[kind]["n"] += 1
            by_kind[kind]["max_diff"] = float("nan")
            continue
        diff = float(np.max(np.abs(bw.astype(np.float32) - qw.astype(np.float32))))
        kind = "model.norm" if k == "model.norm.weight" else k.split(".")[-2]
        s = by_kind[kind]
        s["n"] += 1
        s["max_diff"] = max(s["max_diff"], diff)
        if diff == 0.0:
            s["exact"] += 1
    print(f"  matched {len(matches)} norm tensors")
    print(f"  {'kind':<32} {'n':>4} {'exact-eq':>10} {'worst max-diff':>20}")
    for kind, s in sorted(by_kind.items()):
        print(f"  {kind:<32} {s['n']:>4} {s['exact']:>10} {s['max_diff']:>20.4g}")
    return dict(by_kind)


# -----------------------------------------------------------------------------
# B. Embedding vocab diff
# -----------------------------------------------------------------------------
def section_B_embed_vocab(unp: Path, base: Path) -> dict:
    print("\n== B. embed_tokens vocab diff ==")
    bw = load_tensor(unp, "model.embed_tokens.weight")
    qw = load_tensor(base, "model.embed_tokens.weight")
    out: dict = {}
    if bw is None or qw is None:
        print("  embed_tokens missing on one side")
        return out
    print(f"  Bonsai shape={bw.shape}  Qwen3 shape={qw.shape}")
    out["bonsai_vocab"] = int(bw.shape[0])
    out["base_vocab"]   = int(qw.shape[0])
    out["delta"]        = int(qw.shape[0] - bw.shape[0])
    print(f"  vocab delta: {out['delta']} tokens")
    if bw.shape[0] < qw.shape[0]:
        # Bonsai is a STRICT prefix? Try matching the first bw.shape[0] rows.
        # Compute row-wise cosine for first N to see if rows still align.
        N = bw.shape[0]
        a = bw.astype(np.float32)
        b = qw[:N].astype(np.float32)
        an = np.linalg.norm(a, axis=-1) + 1e-12
        bn = np.linalg.norm(b, axis=-1) + 1e-12
        cos = (np.einsum("ij,ij->i", a, b) / (an * bn))
        out["mean_cos_first_N"] = float(cos.mean())
        out["min_cos_first_N"]  = float(cos.min())
        print(f"  cos(Bonsai_row, Qwen3_row) for first {N} rows: mean={cos.mean():.4f} min={cos.min():.4f}")
        # which dropped rows have the largest |w|?
        dropped = qw[N:].astype(np.float32)
        norms = np.linalg.norm(dropped, axis=-1)
        print(f"  norm stats of the {qw.shape[0] - N} dropped rows: "
              f"mean={float(norms.mean()):.4f} max={float(norms.max()):.4f}")
        out["dropped_norm_mean"] = float(norms.mean())
        out["dropped_norm_max"] = float(norms.max())
    return out


# -----------------------------------------------------------------------------
# C. LM-head presence / tying
# -----------------------------------------------------------------------------
def section_C_lm_head(unp: Path, base: Path) -> dict:
    print("\n== C. LM-head presence / tying ==")
    out: dict = {}
    bk = set(list_tensor_names(unp))
    qk = set(list_tensor_names(base))
    out["bonsai_has_lm_head"] = "lm_head.weight" in bk
    out["base_has_lm_head"]   = "lm_head.weight" in qk
    print(f"  Bonsai-unpacked has lm_head.weight? {out['bonsai_has_lm_head']}")
    print(f"  Qwen3-base       has lm_head.weight? {out['base_has_lm_head']}")
    if out["bonsai_has_lm_head"] and out["base_has_lm_head"]:
        bw = load_tensor(unp, "lm_head.weight")
        qw = load_tensor(base, "lm_head.weight")
        if bw is not None and qw is not None and bw.shape == qw.shape:
            an = np.linalg.norm(bw, axis=-1) + 1e-12
            bn = np.linalg.norm(qw, axis=-1) + 1e-12
            c = float((np.einsum("ij,ij->i", bw, qw) / (an * bn)).mean())
            print(f"  cos(Bonsai lm_head, Qwen3 lm_head) row-mean = {c:.4f}")
            out["lm_head_cos"] = c
    if not out["bonsai_has_lm_head"] and "model.embed_tokens.weight" in bk:
        # tied? compare lm_head (from base) to Bonsai's embed
        bw_emb = load_tensor(unp, "model.embed_tokens.weight")
        if "lm_head.weight" in qk:
            qw_lm = load_tensor(base, "lm_head.weight")
            if bw_emb is not None and qw_lm is not None and bw_emb.shape == qw_lm.shape:
                an = np.linalg.norm(bw_emb, axis=-1) + 1e-12
                bn = np.linalg.norm(qw_lm, axis=-1) + 1e-12
                c = float((np.einsum("ij,ij->i", bw_emb, qw_lm) / (an * bn)).mean())
                print(f"  cos(Bonsai embed_tokens, Qwen3 lm_head) row-mean = {c:.4f}  "
                      f"({'plausible tying' if c > 0.5 else 'unrelated'})")
                out["embed_vs_base_lmhead_cos"] = c
    return out


# -----------------------------------------------------------------------------
# D. Cosine breakdown by projection type
# -----------------------------------------------------------------------------
PROJECTION_KINDS = [
    ("self_attn.q_proj", "Wq"),
    ("self_attn.k_proj", "Wk"),
    ("self_attn.v_proj", "Wv"),
    ("self_attn.o_proj", "Wo"),
    ("mlp.gate_proj", "gate"),
    ("mlp.up_proj",   "up"),
    ("mlp.down_proj", "down"),
]


def row_cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    an = np.linalg.norm(a, axis=-1) + 1e-12
    bn = np.linalg.norm(b, axis=-1) + 1e-12
    return float((np.einsum("ij,ij->i", a, b) / (an * bn)).mean())


def section_D_cosine_breakdown(unp: Path, base: Path) -> dict:
    print("\n== D. cosine(Bonsai-unpacked, Qwen3-base) breakdown by projection type ==")
    bk = set(list_tensor_names(unp))
    qk = set(list_tensor_names(base))
    common = bk & qk
    by_kind: dict[str, list[float]] = {label: [] for _, label in PROJECTION_KINDS}
    for name in sorted(common):
        for needle, label in PROJECTION_KINDS:
            if needle in name and name.endswith(".weight"):
                bw = load_tensor(unp, name)
                qw = load_tensor(base, name)
                if bw is None or qw is None or bw.shape != qw.shape or bw.ndim != 2:
                    continue
                by_kind[label].append(row_cos_mean(bw, qw))
                break
    print(f"  {'kind':<6} {'n':>3} {'cos μ':>8} {'cos σ':>8} {'cos min':>8} {'cos max':>8}")
    out = {}
    for label in [l for _, l in PROJECTION_KINDS]:
        v = np.array(by_kind[label], dtype=np.float64) if by_kind[label] else np.array([])
        if v.size:
            print(f"  {label:<6} {v.size:>3} "
                  f"{v.mean():>8.4f} {v.std():>8.4f} {v.min():>8.4f} {v.max():>8.4f}")
            out[label] = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                          "min": float(v.min()), "max": float(v.max())}
    return out


# -----------------------------------------------------------------------------
# E. Per-head magnitude profile
# -----------------------------------------------------------------------------
def section_E_per_head_magnitude(unp: Path, base: Path, head_dim: int = 128) -> dict:
    print("\n== E. per-head magnitude profile (Wv as the proxy) ==")
    out = {}
    bk = set(list_tensor_names(unp))
    qk = set(list_tensor_names(base))
    layer_indices = sorted(set(int(m.group(1)) for n in bk
                               for m in [re.match(r"model\.layers\.(\d+)\.", n)] if m))
    print(f"  scanning {len(layer_indices)} layers")
    # Just the first 6 layers for brevity
    layers_to_show = layer_indices[: min(6, len(layer_indices))] + layer_indices[-2:]
    for L in sorted(set(layers_to_show)):
        for kind in ["self_attn.v_proj", "self_attn.q_proj"]:
            name = f"model.layers.{L}.{kind}.weight"
            if name not in bk or name not in qk:
                continue
            bw = load_tensor(unp, name)
            qw = load_tensor(base, name)
            if bw is None or qw is None or bw.shape != qw.shape:
                continue
            out_features = bw.shape[0]
            if out_features % head_dim != 0:
                continue
            n_heads = out_features // head_dim
            bh = np.abs(bw.astype(np.float32)).reshape(n_heads, head_dim, -1).mean(axis=(1, 2))
            qh = np.abs(qw.astype(np.float32)).reshape(n_heads, head_dim, -1).mean(axis=(1, 2))
            ratio = bh / (qh + 1e-12)
            print(f"  layer {L:>2} {kind:<18} heads={n_heads}  "
                  f"ratio: μ={ratio.mean():.3f} σ={ratio.std():.3f} "
                  f"range=[{ratio.min():.3f}, {ratio.max():.3f}]")
            out.setdefault(f"L{L}.{kind}", {
                "n_heads": int(n_heads),
                "ratio_mean": float(ratio.mean()),
                "ratio_std":  float(ratio.std()),
                "ratio_min":  float(ratio.min()),
                "ratio_max":  float(ratio.max()),
            })
    return out


# -----------------------------------------------------------------------------
# F. Cosine outliers
# -----------------------------------------------------------------------------
def section_F_outliers(unp: Path, base: Path) -> dict:
    print("\n== F. row-cosine outliers (Bonsai-vs-base) — bottom 5 across all 2D weights ==")
    bk = set(list_tensor_names(unp))
    qk = set(list_tensor_names(base))
    candidates = sorted([n for n in bk & qk if any(p in n for p, _ in PROJECTION_KINDS)
                          and n.endswith(".weight")])
    worst = []  # (cos, name, row_idx)
    for name in candidates:
        bw = load_tensor(unp, name)
        qw = load_tensor(base, name)
        if bw is None or qw is None or bw.shape != qw.shape or bw.ndim != 2:
            continue
        a = bw.astype(np.float32)
        b = qw.astype(np.float32)
        an = np.linalg.norm(a, axis=-1) + 1e-12
        bn = np.linalg.norm(b, axis=-1) + 1e-12
        cos_per_row = np.einsum("ij,ij->i", a, b) / (an * bn)
        # Take this tensor's worst single row
        idx = int(cos_per_row.argmin())
        worst.append((float(cos_per_row[idx]), name, idx))
    worst.sort()
    out = []
    print(f"  {'cos':>7} {'row':>5}  name")
    for cos, name, row in worst[:10]:
        print(f"  {cos:>7.4f} {row:>5}  {name}")
        out.append({"cos": cos, "row": row, "name": name})
    return out


# -----------------------------------------------------------------------------
# G. Scale-distribution shape
# -----------------------------------------------------------------------------
def section_G_scale_shape(gguf: Path, base: Path) -> dict:
    print("\n== G. mean(|x|) / std(x) ratio per 128-block (Bonsai scale shape vs Qwen3) ==")
    print("  Gaussian reference: ratio = sqrt(2/π) ≈ 0.798")
    out = {}
    r = GGUFReader(str(gguf), "r")
    q1 = [t for t in r.tensors if t.tensor_type == GGMLQuantizationType.Q1_0]
    # Pick a small representative subset
    take = []
    for kind, label in PROJECTION_KINDS:
        for t in q1:
            if kind.replace("self_attn.", "attn_").replace("mlp.", "ffn_").replace("_proj", "") in t.name:
                take.append((label, t.name))
                break
    print(f"  showing {len(take)} representative tensors")
    print(f"  {'kind':<6} {'tensor':<40} {'Bonsai s/√(σ²)':>18} {'Qwen3 mean|x|/σ':>18}")
    for label, name in take:
        # Bonsai: s_g (= mean|x| of binary lattice block, all elements have |w|=s, so std(|w|)=0,
        # but std of signed w[block] = s, and mean(|w|) = s. So ratio mean(|x|)/std(x) = 1.)
        # Less interesting for Bonsai. Compute ratio for Qwen3 base.
        from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates
        cand = gguf_to_hf_candidates(name)
        qw = None
        for c in cand:
            qw = load_tensor(base, c)
            if qw is not None:
                break
        if qw is None:
            continue
        flat = qw.reshape(-1).astype(np.float32)
        if flat.size % QK1_0:
            continue
        groups = flat.reshape(-1, QK1_0)
        m = np.mean(np.abs(groups), axis=-1)
        s = np.std(groups, axis=-1) + 1e-12
        ratios_q = m / s

        # Bonsai: by construction within a block, w_i = ±s_g. So |w| = s_g constant,
        # mean(|w|)/std(w) = s/(s) = 1 exactly. Show that for completeness.
        ratios_b = np.full_like(ratios_q, 1.0)

        print(f"  {label:<6} {name:<40} {ratios_b.mean():>18.4f} {ratios_q.mean():>18.4f}")
        out[name] = {"bonsai_ratio_mean": 1.0, "qwen3_ratio_mean": float(ratios_q.mean()),
                     "qwen3_ratio_std": float(ratios_q.std())}
    print()
    print("  Reading guide:")
    print("    Qwen3 ratio ≈ 0.80 means weights are roughly Gaussian within each 128-block.")
    print("    Lower → heavier tails (outliers); higher → more uniform / sub-Gaussian.")
    print("    Bonsai is degenerate (constant |w| per block) so its ratio is 1.0 by construction.")
    return out


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("unpacked")
    ap.add_argument("base")
    args = ap.parse_args()
    g, u, b = Path(args.gguf), Path(args.unpacked), Path(args.base)
    section_A_norm_equality(u, b)
    section_B_embed_vocab(u, b)
    section_C_lm_head(u, b)
    section_D_cosine_breakdown(u, b)
    section_E_per_head_magnitude(u, b)
    section_F_outliers(u, b)
    section_G_scale_shape(g, b)


if __name__ == "__main__":
    main()
