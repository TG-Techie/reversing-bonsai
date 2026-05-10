# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""Generate figures for the Bonsai-1.7B mini report.

Loads the trio (Q1_0 GGUF + Bonsai-unpacked + Qwen3-1.7B base) and writes
PNGs into reports/bonsai-1.7B/figures/. Re-running is idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gguf import GGUFReader, GGMLQuantizationType

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from q1_0 import QK1_0, parse_q1_0
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates
from compare_unpacked_vs_qwen3 import load_tensor as load_st


REPO = THIS.parent
OUT_DIR = REPO / "reports" / "bonsai-1.7B" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GGUF_PATH = REPO / "models" / "q1" / "Bonsai-1.7B-Q1_0.gguf"
UNP_PATH = REPO / "models" / "unpacked" / "model.safetensors"
BASE_PATH = REPO / "models" / "base" / "model-00001-of-00002.safetensors"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_h1_report() -> dict:
    """Parse reports/bonsai-1.7B/05_dequant_vs_unpacked.txt for headline numbers."""
    p = REPO / "reports" / "bonsai-1.7B" / "05_dequant_vs_unpacked.txt"
    txt = p.read_text()
    diffs, lat, sign, scale = [], [], [], []
    for blk in txt.split("\n--"):
        m = re.search(r"max\|deq - unpacked\|\s*=\s*([0-9.eE+\-]+)", blk)
        if m:
            diffs.append(float(m.group(1)))
        m = re.search(r"frac_binary=([\d.]+)", blk)
        if m:
            lat.append(float(m.group(1)))
        m = re.search(r"sign agreement \(nonzero\)\s*=\s*([\d.]+)%", blk)
        if m:
            sign.append(float(m.group(1)))
        m = re.search(r"scale max\|d_q1 - mean\(\|x\|\)\|\s*=\s*([0-9.eE+\-]+)", blk)
        if m:
            scale.append(float(m.group(1)))
    return {
        "max_diff":     np.array(diffs),
        "lattice_frac": np.array(lat),
        "sign_agree":   np.array(sign),
        "scale_diff":   np.array(scale),
    }


def fig1_h1_max_diff(h1: dict) -> Path:
    """Distribution of max|dequant(Q1_0) - unpacked| across 197 tensors."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = h1["max_diff"]
    # log-spaced bins; include zero by clipping smallest non-zero
    nz = d[d > 0]
    if nz.size:
        lo = max(nz.min() / 4, 1e-9)
    else:
        lo = 1e-9
    bins = np.geomspace(lo, max(d.max(), lo * 4), 30)
    ax.hist(np.clip(d, lo, None), bins=bins, color="#4c72b0",
            edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(2 ** -10, color="#dd8452", linestyle="--", linewidth=1,
               label="FP16 ULP at scale 1 (2$^{-10}$)")
    ax.set_xlabel("max |dequant(Q1$_0$) − unpacked| per tensor")
    ax.set_ylabel("# tensors")
    ax.set_title("H1 — every tensor agrees to ≤ 1 FP16 ULP\n"
                 f"({(d == 0).sum()} of {d.size} are bit-identical; worst = {d.max():.3g})")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "fig1_h1_max_diff.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_lattice_and_sign(h1: dict) -> Path:
    """Two side-by-side: lattice frac (≈ 1) and sign agreement (= 100%)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    lat = h1["lattice_frac"]
    ax.hist(lat, bins=np.linspace(min(0.998, lat.min() - 1e-4), 1.0001, 20),
            color="#55a868", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("frac of 128-blocks with one distinct |w|")
    ax.set_ylabel("# tensors")
    ax.set_title(f"binary-lattice frac per tensor\n(global: {lat.mean():.4f})")

    ax = axes[1]
    sign = h1["sign_agree"]
    ax.hist(sign, bins=np.linspace(99.99, 100.001, 10),
            color="#c44e52", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("sign agreement (%)")
    ax.set_title(f"sign agreement per tensor\n(min = {sign.min():.4f}%)")
    fig.suptitle("H1 detail — unpacked file IS the binary lattice", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig2_h1_detail.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_q1_signs(name: str) -> tuple[np.ndarray, np.ndarray]:
    r = GGUFReader(str(GGUF_PATH), "r")
    for t in r.tensors:
        if t.name == name and t.tensor_type == GGMLQuantizationType.Q1_0:
            n = int(np.prod(t.shape))
            raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
            return parse_q1_0(raw, n)
    raise KeyError(name)


def fig3_h3_sign_transitions() -> Path:
    """Histogram of sign transitions per 128-block, overlaid with binomial."""
    # Use ffn_down of block 0 (98,304 blocks) — plenty of data.
    scales, signs = parse_q1_signs("blk.0.ffn_down.weight")
    transitions = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 128) - 0.5
    ax.hist(transitions, bins=bins, density=True, color="#4c72b0",
            edgecolor="white", linewidth=0.3, label="observed (blk.0.ffn_down)")
    # Binomial(127, 0.5) reference
    from math import comb
    k = np.arange(0, 128)
    binom = np.array([comb(127, kk) * 0.5 ** 127 for kk in k])
    ax.plot(k, binom, color="#dd8452", linewidth=1.6, label="Binomial(127, 0.5)")
    ax.set_xlabel("sign transitions per 128-block")
    ax.set_ylabel("density")
    ax.set_title("H3 — signs are statistically random within each block\n"
                 f"(mean={transitions.mean():.2f}, max-entropy expectation = 63.5)")
    ax.set_xlim(20, 105)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "fig3_h3_sign_transitions.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def parse_h2_report() -> dict[int, dict[str, float]]:
    """Parse 03_unpacked_vs_qwen3.txt for cos(identity) by layer.tensor."""
    p = REPO / "reports" / "bonsai-1.7B" / "03_unpacked_vs_qwen3.txt"
    txt = p.read_text()
    out: dict[int, dict[str, float]] = {}
    cur_block = None
    cur_name = None
    for line in txt.splitlines():
        m = re.match(r"^-- (model\.layers\.(\d+)\.\S+)", line)
        if m:
            cur_name = m.group(1)
            cur_block = int(m.group(2))
            continue
        m = re.search(r"identity:\s+rmse=[\d.eE+\-]+\s+cos=([\d.eE+\-]+)", line)
        if m and cur_name is not None and cur_block is not None:
            short = cur_name.split(".", 3)[-1]
            out.setdefault(cur_block, {})[short] = float(m.group(1))
    return out


def fig4_h2_cosine_by_depth(data: dict[int, dict[str, float]]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    blocks = sorted(data.keys())
    series_keys = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                   "self_attn.o_proj.weight",
                   "mlp.gate_proj.weight", "mlp.up_proj.weight",
                   "mlp.down_proj.weight"]
    for k in series_keys:
        ys = [data[b].get(k, np.nan) for b in blocks]
        ax.plot(blocks, ys, marker="o", linewidth=1.4, label=k.replace(".weight", ""))
    ax.axhline(np.sqrt(2 / np.pi), color="black", linestyle="--", linewidth=1)
    ax.text(blocks[-1], np.sqrt(2 / np.pi) + 0.012,
            f"sqrt(2/π) ≈ {np.sqrt(2/np.pi):.3f}\n(pure sign-quant of Gaussian Qwen3)",
            ha="right", fontsize=8, color="black")
    ax.set_xticks(blocks)
    ax.set_xlabel("Qwen3 / Bonsai layer index")
    ax.set_ylabel("identity row cosine vs Qwen3-base")
    ax.set_ylim(0.30, 0.85)
    ax.set_title("H2 — Bonsai's signs aren't Qwen3's signs (cos < 0.8)\n"
                 "and the gap widens with depth")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "fig4_h2_cosine_by_depth.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def parse_magnitudes() -> dict:
    """Parse 06_magnitudes.txt: aggregates by block."""
    p = REPO / "reports" / "bonsai-1.7B" / "06_magnitudes.txt"
    text = p.read_text()
    blocks_seen = []
    cur_filter = None
    aggs: dict[int, dict[str, float]] = {}
    in_agg = False
    for line in text.splitlines():
        m = re.search(r"filter regex:\s+blk\.(\d+)", line)
        if m:
            cur_filter = int(m.group(1))
            blocks_seen.append(cur_filter)
            in_agg = False
            continue
        if "AGGREGATE" in line:
            in_agg = True
            aggs[cur_filter] = {}
            continue
        if in_agg and cur_filter is not None:
            m = re.match(r"\s+(\S.*?)\s+mean=([+\-\d.]+)\s+min=([+\-\d.]+)\s+max=([+\-\d.]+)", line)
            if m:
                key = m.group(1).strip()
                aggs[cur_filter][key] = {
                    "mean": float(m.group(2)),
                    "min":  float(m.group(3)),
                    "max":  float(m.group(4)),
                }
    return aggs


def fig5_magnitudes_by_depth(aggs: dict) -> Path:
    blocks = sorted(aggs.keys())
    keys = [
        ("corr s_g vs base mean(|w|)",  "scale s_g vs base mean(|w|)"),
        ("corr per-row mean|w|",        "per-row mean(|w|)"),
        ("corr per-col mean|w|",        "per-col mean(|w|)"),
    ]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (k, lbl), c in zip(keys, colors):
        means = [aggs[b][k]["mean"] for b in blocks]
        mins  = [aggs[b][k]["min"]  for b in blocks]
        maxs  = [aggs[b][k]["max"]  for b in blocks]
        ax.plot(blocks, means, "o-", linewidth=1.6, color=c, label=lbl)
        ax.fill_between(blocks, mins, maxs, color=c, alpha=0.15)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xticks(blocks)
    ax.set_xlabel("layer index")
    ax.set_ylabel("Pearson correlation, Bonsai vs Qwen3-base")
    ax.set_title("Magnitude follow-up — per-row preserved, per-col washed out\n"
                 "and overall correlation drops with depth")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    out = OUT_DIR / "fig5_magnitudes_by_depth.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig6_abs_distribution() -> Path:
    """Histogram of |w| for Bonsai-unpacked vs Qwen3 base, one tensor."""
    name_q1 = "blk.0.attn_output.weight"
    name_hf = gguf_to_hf_candidates(name_q1)[0]
    bons = load_st(UNP_PATH, name_hf)
    base = load_st(BASE_PATH, name_hf)
    bons_abs = np.abs(bons.astype(np.float32)).ravel()
    base_abs = np.abs(base.astype(np.float32)).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=False)
    bins = np.linspace(0, max(bons_abs.max(), base_abs.max()) * 1.05, 80)
    ax = axes[0]
    ax.hist(base_abs, bins=bins, density=True, alpha=0.55, color="#4c72b0",
            label=f"Qwen3 base (μ={base_abs.mean():.3f})", edgecolor="none")
    ax.hist(bons_abs, bins=bins, density=True, alpha=0.55, color="#dd8452",
            label=f"Bonsai unpacked (μ={bons_abs.mean():.3f})", edgecolor="none")
    ax.set_xlabel("|w|")
    ax.set_ylabel("density")
    ax.set_title(f"|w| distribution, {name_hf}\n(linear scale)")
    ax.legend(frameon=False, fontsize=8)

    # Right panel: log-y, to see Bonsai's discrete spikes
    ax = axes[1]
    ax.hist(base_abs, bins=bins, density=True, alpha=0.55, color="#4c72b0",
            label="Qwen3 base", edgecolor="none")
    ax.hist(bons_abs, bins=bins, density=True, alpha=0.55, color="#dd8452",
            label="Bonsai unpacked", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("|w|")
    ax.set_title("same, log-y\n(Bonsai's spikes are individual scale values)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Bonsai is discrete (~ a few hundred scale values + signs); "
                 "Qwen3 is continuous", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig6_abs_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_scale_vs_base_scatter() -> Path:
    """Scatter of per-block (s_g vs base group mean|w|) for one tensor."""
    name_q1 = "blk.0.attn_output.weight"
    name_hf = gguf_to_hf_candidates(name_q1)[0]
    scales, _ = parse_q1_signs(name_q1)
    base = load_st(BASE_PATH, name_hf)
    base_groups = np.abs(base.astype(np.float32).reshape(-1, QK1_0)).mean(axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # Subsample for plotting speed
    n_plot = min(8000, scales.size)
    idx = np.random.default_rng(0).choice(scales.size, size=n_plot, replace=False)
    ax.scatter(base_groups[idx], scales[idx], s=4, alpha=0.25, color="#4c72b0",
               edgecolors="none")
    lo = 0
    hi = max(base_groups.max(), scales.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="y = x (s = base mean)")
    ax.plot([lo, hi], [lo, 2 * hi], "--", color="#dd8452", linewidth=1, label="y = 2x")
    # Pearson
    a = base_groups[idx] - base_groups[idx].mean()
    b = scales[idx] - scales[idx].mean()
    r = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Qwen3 base group mean(|w|)")
    ax.set_ylabel("Bonsai per-block scale s$_g$")
    ax.set_title(f"per-block scatter, {name_hf}\n(r = {r:.3f}, n_blocks = {scales.size})")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "fig7_scale_vs_base_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    h1 = parse_h1_report()
    h2 = parse_h2_report()
    mags = parse_magnitudes()

    figs = [
        fig1_h1_max_diff(h1),
        fig2_lattice_and_sign(h1),
        fig3_h3_sign_transitions(),
        fig4_h2_cosine_by_depth(h2),
        fig5_magnitudes_by_depth(mags),
        fig6_abs_distribution(),
        fig7_scale_vs_base_scatter(),
    ]
    print("Wrote:")
    for f in figs:
        print(" ", f.relative_to(REPO))


if __name__ == "__main__":
    main()
