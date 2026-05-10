"""Per-attention-head per-block scale analysis at 8B.

Bonsai-8B has 32 query heads × 128 head_dim, 8 KV heads × 128 head_dim,
hidden_size 4096. The shape conventions (HF orientation):

  q_proj.weight: (out=4096, in=4096) where out = 32 heads × 128 head_dim
  k_proj.weight: (out=1024, in=4096) where out = 8 heads × 128 head_dim
  v_proj.weight: (out=1024, in=4096) where out = 8 heads × 128 head_dim

For Q1_0_g128, blocks are 128 along the input (last) axis, so per-row
there are 4096/128 = 32 blocks. Each row's per-block scales describe
how the corresponding output channel reads its inputs. Reshaping the
output dim into (heads, head_dim) lets us ask: does the technique
treat heads uniformly, or do specific heads get systematically
amplified / damped?

Test: for each layer and each {q, k, v} projection, group rows by
attention-head index. Compute per-head mean and std of the per-block
scales. Then ask:

  H1: are heads' mean scales tightly clustered, or some heads
      systematically amplified?
  H2: does the per-head scale-mean evolve with depth? (e.g. maybe
      layer-1 head-3 is a high-scale head and stays so, suggesting
      fixed head-level capacity allocation).

Output: ASCII tables per projection type.

Reads only the GGUF; peak RAM is one tensor at a time.
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


# Architecture knowledge: hard-coded for 8B.
HEAD_DIM = 128
N_Q_HEADS = 32
N_KV_HEADS = 8


def main(size: str = "8B") -> None:
    gguf_path = ROOT / f"models/bonsai/{size}/gguf/Bonsai-{size}-Q1_0.gguf"
    print(f"size={size}  gguf={gguf_path}", flush=True)

    r = GGUFReader(str(gguf_path), "r")
    head_scale_means = {}  # (layer, kind, head) -> mean(per-block scale across that head's rows)
    head_scale_stds = {}
    rows_per_kind = {"q": HEAD_DIM * N_Q_HEADS, "k": HEAD_DIM * N_KV_HEADS, "v": HEAD_DIM * N_KV_HEADS}

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
        else: continue

        n_heads = N_Q_HEADS if kind == "q" else N_KV_HEADS
        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        n_elems = int(np.prod(list(t.shape)))
        scales, _ = parse_q1_0(raw, n_elems)
        gguf_shape = list(t.shape)
        hf_out = gguf_shape[1]
        hf_in = gguf_shape[0]
        n_blocks_per_row = hf_in // 128
        sg = scales.reshape(hf_out, n_blocks_per_row).astype(np.float32)
        # Verify expected shape
        assert hf_out == n_heads * HEAD_DIM, f"expected {n_heads}*{HEAD_DIM} out, got {hf_out}"
        # Group by head: rows[h*HEAD_DIM:(h+1)*HEAD_DIM] are head h's rows
        head_rows = sg.reshape(n_heads, HEAD_DIM, n_blocks_per_row)
        for h in range(n_heads):
            flat = head_rows[h].reshape(-1)
            head_scale_means[(layer, kind, h)] = float(flat.mean())
            head_scale_stds[(layer, kind, h)] = float(flat.std())

    del r
    print(f"loaded {len(head_scale_means)} (layer, kind, head) entries", flush=True)

    LAYERS = sorted(set(L for L, _, _ in head_scale_means.keys()))

    # H1: per-tensor head-to-head variation. Within one (layer, kind), is the
    # head-mean scale tightly clustered or wide?
    print(f"\n=== H1: per-tensor head-mean scale spread ===", flush=True)
    print("std-across-heads / mean-across-heads. Small = uniform across heads. Large = some heads are amplified vs others.", flush=True)
    rows = []
    for L in LAYERS:
        for kind in ["q", "k", "v"]:
            n_heads = N_Q_HEADS if kind == "q" else N_KV_HEADS
            head_means = np.array([head_scale_means[(L, kind, h)] for h in range(n_heads) if (L, kind, h) in head_scale_means])
            if head_means.size == 0: continue
            mu = head_means.mean()
            sigma = head_means.std()
            cv = sigma / max(mu, 1e-12)
            rows.append({
                "L": L,
                "kind": kind,
                "n_heads": int(head_means.size),
                "head_mean_mu": f"{mu:.5f}",
                "head_mean_sigma": f"{sigma:.5f}",
                "cv": f"{cv:.4f}",
            })

    # Print one table per kind
    for kind in ["q", "k", "v"]:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub: continue
        print(f"\n--- per-head spread, kind={kind} ---", flush=True)
        print(fmt_table(
            ["L", "n_heads", "head_mean_mu", "head_mean_sigma", "cv"],
            [[s["L"], s["n_heads"], s["head_mean_mu"], s["head_mean_sigma"], s["cv"]] for s in sub],
        ))

    # H2: are the same heads "loud" at every layer? Compute head-mean rank-correlation
    # between consecutive layers.
    print(f"\n=== H2: same-head identity across layers (rank correlation of head-mean scales) ===", flush=True)
    print("Spearman near 0: each layer rearranges 'loud' heads. Spearman near 1: head identity preserved across depth.", flush=True)
    from scipy.stats import spearmanr
    h2_rows = []
    for kind in ["q", "k", "v"]:
        n_heads = N_Q_HEADS if kind == "q" else N_KV_HEADS
        spearmans = []
        for L in LAYERS:
            if (L + 1) not in LAYERS: continue
            a = np.array([head_scale_means.get((L, kind, h), 0.0) for h in range(n_heads)])
            b = np.array([head_scale_means.get((L + 1, kind, h), 0.0) for h in range(n_heads)])
            sp, _ = spearmanr(a, b)
            spearmans.append(float(sp))
        if spearmans:
            h2_rows.append({
                "kind": kind,
                "n_pairs": len(spearmans),
                "spearman_mean": f"{np.mean(spearmans):+.3f}",
                "spearman_std": f"{np.std(spearmans):.3f}",
                "min": f"{min(spearmans):+.3f}",
                "max": f"{max(spearmans):+.3f}",
            })
    print(fmt_table(
        ["kind", "n_pairs", "spearman_mean", "spearman_std", "min", "max"],
        [[r["kind"], r["n_pairs"], r["spearman_mean"], r["spearman_std"], r["min"], r["max"]] for r in h2_rows],
    ))

    # H2-extreme: is head 0 ALWAYS the loudest, or does it vary?
    print(f"\n=== H2-extreme: which head is the loudest at each layer? ===", flush=True)
    for kind in ["q", "k", "v"]:
        n_heads = N_Q_HEADS if kind == "q" else N_KV_HEADS
        loud_head_per_layer = []
        for L in LAYERS:
            scores = [head_scale_means.get((L, kind, h), 0.0) for h in range(n_heads)]
            loud_head_per_layer.append(int(np.argmax(scores)))
        # Print a histogram of which head is the loudest across layers
        from collections import Counter
        c = Counter(loud_head_per_layer)
        top_3 = c.most_common(3)
        total = sum(c.values())
        print(f"  kind={kind}: head 'loudest at how many layers?' top 3: {top_3}  (out of {total} layers)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "8B")
