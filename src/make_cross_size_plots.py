# SPDX-License-Identifier: MIT
"""Generate cross-size visualization figures from the *.txt reports under
reports/local-{1.7B,4B,8B}/. Produces PNGs into reports/figures/."""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def parse_sign_disagreement_per_depth(path: Path) -> dict[int, float]:
    """From sign_disagreement / ptq_baseline output, extract per-depth mean.
    Both formats start a line with `   <depth>   <mean>   ...`; we only need
    the first two columns and tolerate trailing extras."""
    out = {}
    text = path.read_text()
    for m in re.finditer(r"^\s*(\d+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+", text, re.MULTILINE):
        # avoid matching tensor-name lines that happen to start with a number
        out[int(m.group(1))] = float(m.group(2))
    return out


def parse_filtered_sign(path: Path) -> float:
    """From sign_disagreement output filtered to a single layer, extract OVERALL mean."""
    text = path.read_text()
    m = re.search(r"OVERALL: mean=([0-9.]+)", text)
    return float(m.group(1)) if m else float("nan")


def parse_h2_layer_cosines(path: Path) -> list[float]:
    """Extract identity row cosines from compare_unpacked_vs_qwen3 output (single layer filter)."""
    text = path.read_text()
    return [float(m.group(1)) for m in re.finditer(r"identity:\s+rmse=[0-9.eE+-]+\s+cos=([0-9.]+)", text)]


def parse_mag_aggregate(path: Path) -> dict[str, float]:
    """Extract `corr s_g vs base mean(|w|)` aggregate from compare_magnitudes output."""
    text = path.read_text()
    out = {}
    pat = (
        r"corr s_g vs base mean\(\|w\|\)\s+mean=([+-][0-9.]+)\s+min=([+-][0-9.]+)\s+max=([+-][0-9.]+)"
    )
    m = re.search(pat, text)
    if m:
        out["s_g_corr_mean"] = float(m.group(1))
        out["s_g_corr_min"] = float(m.group(2))
        out["s_g_corr_max"] = float(m.group(3))
    pat2 = r"corr per-row mean\|w\|\s+mean=([+-][0-9.]+)"
    m = re.search(pat2, text)
    if m:
        out["per_row_corr"] = float(m.group(1))
    pat3 = r"corr per-col mean\|w\|\s+mean=([+-][0-9.]+)"
    m = re.search(pat3, text)
    if m:
        out["per_col_corr"] = float(m.group(1))
    pat4 = r"ratio s_g / base_mean \(median per tensor\)\s+mean=([+-][0-9.]+)"
    m = re.search(pat4, text)
    if m:
        out["loud_ratio"] = float(m.group(1))
    return out


# === Figure 1: Sign disagreement vs depth, all three sizes ===
def fig_sign_disagreement():
    fig, ax = plt.subplots(figsize=(10, 5))
    # 1.7B and 4B have all-layer reports
    for size, fname, color in [
        ("Bonsai-1.7B", "reports/local-1.7B/08_sign_disagreement.txt", "tab:blue"),
        ("Bonsai-4B", "reports/local-4B/08_sign_disagreement.txt", "tab:orange"),
    ]:
        depths = parse_sign_disagreement_per_depth(REPO / fname)
        if not depths:
            continue
        xs = sorted(depths)
        ys = [depths[d] for d in xs]
        # Normalize x to [0, 1] for cross-size overlay
        xs_norm = [d / max(xs) for d in xs]
        ax.plot(xs_norm, ys, marker="o", label=f"{size} ({len(xs)} layers)", color=color, alpha=0.85)
    # 8B has 7 filtered layers
    eight_b = {}
    for L in [0, 6, 12, 18, 24, 30, 35]:
        p = REPO / f"reports/local-8B/08_sign_layer{L}.txt"
        if p.exists() and p.stat().st_size > 0:
            eight_b[L] = parse_filtered_sign(p)
    if eight_b:
        xs = sorted(eight_b)
        ys = [eight_b[d] for d in xs]
        xs_norm = [d / max(xs) for d in xs]
        ax.plot(xs_norm, ys, marker="s", label=f"Bonsai-8B ({len(xs)} layer samples)", color="tab:green")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random (0.5)")
    ax.axhline(0.101, color="black", linestyle="--", linewidth=1, label=r"PTQ Gaussian floor: $\sqrt{2/\pi}$ ≈ 0.798 cos → 0.101 flips")
    ax.set_xlabel("Layer depth (normalized 0=first → 1=last)")
    ax.set_ylabel("Fraction of weights with sign disagreement vs Qwen3-base")
    ax.set_title("Bonsai vs Qwen3-base: sign-disagreement rate across depth, three sizes")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig01_sign_disagreement_cross_size.png", dpi=120)
    plt.close()
    print("wrote fig01_sign_disagreement_cross_size.png")


# === Figure 2: Identity row cosine vs depth ===
def fig_h2_cosine():
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = [
        ("Bonsai-1.7B", "reports/local-1.7B", [0, 13, 27], "tab:blue"),
        ("Bonsai-4B", "reports/local-4B", [0, 8, 17, 26, 35], "tab:orange"),
        ("Bonsai-8B", "reports/local-8B", [0, 6, 12, 18, 24, 30, 35], "tab:green"),
    ]
    for label, dir, layers, color in sizes:
        all_layers_xs, all_layers_means, all_layers_lo, all_layers_hi = [], [], [], []
        Lmax = max(layers)
        for L in layers:
            p = REPO / f"{dir}/03_h2_layer{L}.txt"
            if not p.exists():
                continue
            cos = parse_h2_layer_cosines(p)
            if not cos:
                continue
            all_layers_xs.append(L / Lmax)
            all_layers_means.append(np.mean(cos))
            all_layers_lo.append(np.min(cos))
            all_layers_hi.append(np.max(cos))
        ax.plot(all_layers_xs, all_layers_means, marker="o", color=color, label=f"{label} (mean across 7 projs)")
        ax.fill_between(all_layers_xs, all_layers_lo, all_layers_hi, alpha=0.15, color=color)
    ax.axhline(0.798, color="black", linestyle="--", linewidth=1, label=r"$\sqrt{2/\pi}$ (PTQ Gaussian floor)")
    ax.set_xlabel("Layer depth (normalized)")
    ax.set_ylabel("Identity row cosine (Bonsai vs Qwen3-base)")
    ax.set_title("Bonsai row-cosine vs base, across depth and size — band = min/max across 7 projections")
    ax.legend(loc="lower left")
    ax.set_ylim(0.2, 0.85)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig02_h2_cosine_cross_size.png", dpi=120)
    plt.close()
    print("wrote fig02_h2_cosine_cross_size.png")


# === Figure 3: s_g correlation with base group mean, across depth ===
def fig_sg_correlation():
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = [
        ("Bonsai-1.7B", "reports/local-1.7B", [0, 13, 27], "tab:blue"),
        ("Bonsai-4B", "reports/local-4B", [0, 17, 35], "tab:orange"),
        ("Bonsai-8B", "reports/local-8B", [0, 6, 12, 18, 24, 30, 35], "tab:green"),
    ]
    for label, dir, layers, color in sizes:
        xs, ys = [], []
        Lmax = max(layers)
        for L in layers:
            p = REPO / f"{dir}/06_mag_layer{L}.txt"
            if not p.exists():
                continue
            agg = parse_mag_aggregate(p)
            if "s_g_corr_mean" not in agg:
                continue
            xs.append(L / Lmax)
            ys.append(agg["s_g_corr_mean"])
        if xs:
            ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="perfect (= base group mean)")
    ax.set_xlabel("Layer depth (normalized)")
    ax.set_ylabel("Pearson(s_g, base group mean(|w|))")
    ax.set_title("Per-block scale correlation with Qwen3-base group magnitude, across depth")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig03_sg_correlation_cross_size.png", dpi=120)
    plt.close()
    print("wrote fig03_sg_correlation_cross_size.png")


# === Figure 4: PTQ baseline cosine — per-layer + theoretical floor ===
def fig_ptq_baseline():
    fig, ax = plt.subplots(figsize=(10, 5))
    for size, fname, color in [
        ("Qwen3-1.7B PTQ", "reports/local-1.7B/09_ptq_baseline.txt", "tab:blue"),
        ("Qwen3-4B PTQ", "reports/local-4B/09_ptq_baseline.txt", "tab:orange"),
        ("Qwen3-8B PTQ", "reports/local-8B/09_ptq_baseline.txt", "tab:green"),
    ]:
        p = REPO / fname
        if not p.exists():
            continue
        depths = parse_sign_disagreement_per_depth(p)  # same line format
        if not depths:
            continue
        xs = sorted(depths)
        ys = [depths[d] for d in xs]
        xs_norm = [d / max(xs) for d in xs]
        ax.plot(xs_norm, ys, marker="o", color=color, label=size, alpha=0.8)
    ax.axhline(0.7979, color="black", linestyle="--", linewidth=1, label=r"$\sqrt{2/\pi}$ (Gaussian prediction)")
    ax.set_xlabel("Layer depth (normalized)")
    ax.set_ylabel("PTQ row cosine (Q1_0 sign-quant of Qwen3 vs Qwen3)")
    ax.set_title("PTQ baseline: format-induced loss is approximately depth-flat, near Gaussian prediction")
    ax.legend(loc="lower left")
    ax.set_ylim(0.5, 0.85)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig04_ptq_baseline_cross_size.png", dpi=120)
    plt.close()
    print("wrote fig04_ptq_baseline_cross_size.png")


if __name__ == "__main__":
    fig_sign_disagreement()
    fig_h2_cosine()
    fig_sg_correlation()
    fig_ptq_baseline()
