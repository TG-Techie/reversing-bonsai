"""Memory-streaming full-network formula audit. Parameterised by size.

Walk each base shard once. For each tensor in the shard, dequantize the
corresponding Bonsai Q1_0 from the GGUF on demand, compute formula match,
free everything. Peak memory: one tensor pair at a time.

Usage:
  uv run python scripts/streaming_formula_audit.py 4B
  uv run python scripts/streaming_formula_audit.py 8B
"""
import sys
import numpy as np
import gc
import re
from pathlib import Path
from safetensors import safe_open
from gguf import GGUFReader, GGMLQuantizationType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates
from q1_0 import parse_q1_0


def main(size: str) -> None:
    gguf_path = ROOT / f"models/bonsai/{size}/gguf/Bonsai-{size}-Q1_0.gguf"
    base_dir = ROOT / f"models/bonsai/{size}/base"

    print(f"size={size}", flush=True)
    print(f"gguf={gguf_path}", flush=True)
    print(f"base={base_dir}", flush=True)

    print("Indexing GGUF tensors...", flush=True)
    r = GGUFReader(str(gguf_path), "r")
    hf_to_tensor = {}
    for t in r.tensors:
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            continue
        cands = gguf_to_hf_candidates(t.name)
        hf = cands[0] if cands else t.name
        hf_to_tensor[hf] = t
    print(f"  {len(hf_to_tensor)} Q1_0 tensors indexed", flush=True)

    results = []
    for shard in sorted(base_dir.glob("*.safetensors")):
        print(f"\nWalking {shard.name}...", flush=True)
        with safe_open(str(shard), framework="pt") as h:
            for k in h.keys():
                if k not in hf_to_tensor:
                    continue
                try:
                    W_b = h.get_tensor(k).float().numpy()
                except Exception as e:
                    print(f"  skip {k}: {e}", flush=True)
                    continue
                if W_b.ndim != 2:
                    del W_b
                    continue

                t = hf_to_tensor[k]
                raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
                n_elems = int(np.prod(list(t.shape)))
                scales, signs = parse_q1_0(raw, n_elems)
                flat = (signs.astype(np.float32) * scales[:, None]).reshape(-1)
                hf_shape = list(reversed(list(t.shape)))
                W_bonsai = flat.reshape(hf_shape)
                del raw, scales, signs, flat

                nrows = min(W_b.shape[0], W_bonsai.shape[0])
                W_b = W_b[:nrows]
                W_bonsai = W_bonsai[:nrows]
                if W_b.shape != W_bonsai.shape or W_b.shape[1] % 128 != 0:
                    del W_b, W_bonsai
                    continue

                nblk = W_b.shape[1] // 128
                g_b = W_b.reshape(nrows, nblk, 128)
                g_bonsai = W_bonsai.reshape(nrows, nblk, 128)

                s_formula = np.abs(g_b).mean(axis=-1, keepdims=True)
                s_bonsai = np.abs(g_bonsai).mean(axis=-1, keepdims=True)
                formula_w = np.sign(g_b) * s_formula

                nz = np.abs(W_bonsai) > 0
                sign_match = ((np.sign(W_bonsai) == np.sign(W_b)) & nz).sum() / nz.sum()
                scale_match = float((np.abs(s_bonsai - s_formula) < 1e-3 * (s_formula + 1e-9)).mean())
                byte_match = float((np.abs(g_bonsai - formula_w) < 1e-3).mean())

                L = re.search(r"layers\.(\d+)\.", k)
                layer = int(L.group(1)) if L else -1
                if "q_proj" in k: kind = "q"
                elif "k_proj" in k: kind = "k"
                elif "v_proj" in k: kind = "v"
                elif "o_proj" in k: kind = "o"
                elif "mlp.gate" in k: kind = "gate"
                elif "mlp.up" in k: kind = "up"
                elif "mlp.down" in k: kind = "down"
                elif "embed" in k: kind = "embed"
                elif "lm_head" in k: kind = "lm_head"
                else: kind = "?"
                results.append((layer, kind, float(sign_match), scale_match, byte_match))
                print(
                    f"  L{layer:3d} {kind:8s}  sign={sign_match:.4f}  scale={scale_match:.4f}  byte={byte_match:.4f}",
                    flush=True,
                )
                del W_b, W_bonsai, g_b, g_bonsai, s_formula, s_bonsai, formula_w, nz
                gc.collect()

    import collections
    print(f"\n=== per-projection-type summary across all {len(results)} tensors ===", flush=True)
    by_kind = collections.defaultdict(list)
    for L, kind, sm, scm, bm in results:
        by_kind[kind].append((L, sm, scm, bm))
    print(
        f'{"type":>10s}  {"n":>4s}  {"sign_mean":>9s}  {"sign_min":>8s}  {"sign_max":>8s}  {"scale_mean":>10s}  {"byte_mean":>9s}'
    )
    for kind in ["embed", "lm_head", "q", "k", "v", "o", "gate", "up", "down"]:
        if kind not in by_kind:
            continue
        arr = by_kind[kind]
        sm = np.array([s for _, s, _, _ in arr])
        scm = np.array([s for _, _, s, _ in arr])
        bm = np.array([b for _, _, _, b in arr])
        print(
            f"  {kind:>8s}  {len(arr):>4d}  {sm.mean():>9.4f}  {sm.min():>8.4f}  {sm.max():>8.4f}  {scm.mean():>10.4f}  {bm.mean():>9.4f}"
        )

    print(f"\n=== sign-match by layer (rows) and projection type (cols) ===", flush=True)
    LAYERS = sorted(set(L for L, _, _, _, _ in results if L >= 0))
    print(f"L     " + " ".join(f"{k:>6s}" for k in ["q", "k", "v", "o", "gate", "up", "down"]))
    by_lk = {(L, k): sm for L, k, sm, _, _ in results}
    for L in LAYERS:
        row = [
            f'{by_lk.get((L, k), 0):.3f}' if (L, k) in by_lk else "   -"
            for k in ["q", "k", "v", "o", "gate", "up", "down"]
        ]
        print(f"{L:3d}   " + "  ".join(f"{x:>5s}" for x in row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "8B")
