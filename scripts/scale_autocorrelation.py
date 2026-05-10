"""Per-layer scale-distribution evolution at 8B.

Two tests, both reading per-128-block scales `s_g` directly from the GGUF
(no dequantisation needed):

T1. Per-layer scale-variance evolution: for each projection type
    (q/k/v/o/gate/up/down), how does the std and entropy of the per-
    block scale distribution change with depth? OBC-style accumulated
    error predicts increasing variability with depth (more accumulated
    distortion → more block-by-block variation). Independent per-layer
    SGD predicts roughly flat.

T2. Cross-layer matched-position correlation (cautious): for matched
    (row_idx, block_idx) positions across consecutive layers, what's
    the Pearson? If near zero, the technique optimises each layer
    independently. If positive, either layers share initialisation
    or there's a tensor-level distributional similarity that survives
    to per-position level.

Outputs ASCII tables. Script reads only the GGUF.
"""
import sys
import numpy as np
import re
from pathlib import Path
from gguf import GGUFReader, GGMLQuantizationType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates
from q1_0 import parse_q1_0
from asciitable import fmt_table


def main(size: str = "8B") -> None:
    gguf_path = ROOT / f"models/bonsai/{size}/gguf/Bonsai-{size}-Q1_0.gguf"
    print(f"size={size}  gguf={gguf_path}", flush=True)

    r = GGUFReader(str(gguf_path), "r")
    scales_by = {}  # (layer, kind) -> per-block scale array (out, n_blocks_per_row)
    for t in r.tensors:
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            continue
        cands = gguf_to_hf_candidates(t.name)
        hf = cands[0] if cands else t.name
        L = re.search(r"layers\.(\d+)\.", hf)
        if not L:
            continue
        layer = int(L.group(1))
        if "q_proj" in hf: kind = "q"
        elif "k_proj" in hf: kind = "k"
        elif "v_proj" in hf: kind = "v"
        elif "o_proj" in hf: kind = "o"
        elif "mlp.gate" in hf: kind = "gate"
        elif "mlp.up" in hf: kind = "up"
        elif "mlp.down" in hf: kind = "down"
        else: continue

        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        n_elems = int(np.prod(list(t.shape)))
        scales, _ = parse_q1_0(raw, n_elems)
        gguf_shape = list(t.shape)
        hf_out = gguf_shape[1]
        hf_in = gguf_shape[0]
        scale_grid = scales.reshape(hf_out, hf_in // 128).astype(np.float32)
        scales_by[(layer, kind)] = scale_grid
    del r
    print(f"loaded {len(scales_by)} (layer, projection) pairs", flush=True)

    LAYERS = sorted(set(l for l, _ in scales_by.keys()))
    KINDS = ["q", "k", "v", "o", "gate", "up", "down"]

    # T1: per-layer scale stats (mean, std, std/mean cv, max, ratio max/mean)
    print(f"\n=== T1: per-layer per-type scale-distribution stats ===", flush=True)
    print("std_per_layer should rise with depth if OBC-style accumulated-error story holds.", flush=True)
    rows = []
    for kind in KINDS:
        for L in LAYERS:
            if (L, kind) not in scales_by:
                continue
            s = scales_by[(L, kind)].reshape(-1)
            rows.append({
                "type": kind,
                "L": L,
                "mean": f"{s.mean():.5f}",
                "std": f"{s.std():.5f}",
                "cv(std/mean)": f"{s.std() / max(s.mean(), 1e-9):.4f}",
                "max/mean": f"{s.max() / max(s.mean(), 1e-9):.2f}",
            })
    # Print one table per kind for readability
    for kind in KINDS:
        sub = [r for r in rows if r["type"] == kind]
        if not sub: continue
        print(f"\n--- type = {kind} ---", flush=True)
        print(fmt_table(
            ["L", "mean", "std", "cv(std/mean)", "max/mean"],
            [[s["L"], s["mean"], s["std"], s["cv(std/mean)"], s["max/mean"]] for s in sub],
        ))

    # T1 condensed: per-type early-vs-late comparison
    print(f"\n=== T1 condensed: early-vs-late comparison per type ===", flush=True)
    print("'early' = layers 0-3, 'late' = last 4 layers.", flush=True)
    cond = []
    for kind in KINDS:
        early_std = []
        late_std = []
        for L in LAYERS:
            if (L, kind) not in scales_by: continue
            s = scales_by[(L, kind)].reshape(-1)
            cv = s.std() / max(s.mean(), 1e-9)
            if L <= 3:
                early_std.append(cv)
            elif L >= max(LAYERS) - 3:
                late_std.append(cv)
        if early_std and late_std:
            cond.append({
                "type": kind,
                "early_cv_mean": f"{np.mean(early_std):.4f}",
                "late_cv_mean": f"{np.mean(late_std):.4f}",
                "delta": f"{np.mean(late_std) - np.mean(early_std):+.4f}",
                "late/early": f"{np.mean(late_std) / max(np.mean(early_std), 1e-9):.3f}",
            })
    print(fmt_table(
        ["type", "early_cv_mean", "late_cv_mean", "delta", "late/early"],
        [[r["type"], r["early_cv_mean"], r["late_cv_mean"], r["delta"], r["late/early"]] for r in cond],
    ))

    # T2: matched-position autocorrelation (cautious)
    from scipy.stats import spearmanr
    print(f"\n=== T2: cross-layer (L, L+1) matched-position correlation ===", flush=True)
    print("Position = (row_idx, block_idx). Pearson near 0 = independent layers; positive = correlated.", flush=True)
    print("Caveat: row_idx doesn't carry layer-to-layer semantics, so correlation here is tensor-distributional, not OBC-specific.", flush=True)
    t2_rows = []
    for kind in KINDS:
        pears = []
        for L in LAYERS:
            if (L, kind) not in scales_by or (L + 1, kind) not in scales_by:
                continue
            a = scales_by[(L, kind)]
            b = scales_by[(L + 1, kind)]
            if a.shape != b.shape:
                continue
            af = a.reshape(-1)
            bf = b.reshape(-1)
            p = float(np.corrcoef(af, bf)[0, 1])
            pears.append(p)
        if pears:
            t2_rows.append({
                "type": kind,
                "n_pairs": len(pears),
                "pearson_mean": f"{np.mean(pears):+.3f}",
                "pearson_std": f"{np.std(pears):.3f}",
                "min": f"{min(pears):+.3f}",
                "max": f"{max(pears):+.3f}",
            })
    print(fmt_table(
        ["type", "n_pairs", "pearson_mean", "pearson_std", "min", "max"],
        [[r["type"], r["n_pairs"], r["pearson_mean"], r["pearson_std"], r["min"], r["max"]] for r in t2_rows],
    ))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "8B")
