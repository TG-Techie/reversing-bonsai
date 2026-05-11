"""Independent verification of RECIPE_HINTS.md v4 claims at 8B.

Tests:
 1. Sign-match per projection type at L0 and L17 (q/k/v/o/gate/up/down).
    v4 claim: ordering v > down > o > k > q > gate ≈ up.
 2. Magnitude-graded sign flips at L0.q_proj.
    v4 claim: d1 ~0.47, d10 ~0.025.
 3. Per-block scale != mean(|w_teacher|_g): 0% byte-equal across 3 tensors.
 4. Per-row Pearson(per-row mean s_g, per-row mean |w_teacher|).
    v4 claim: attn ~0.8, MLP ~0.4.
"""

from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gguf import GGUFReader
from q1_0 import parse_q1_0
from safetensors.torch import safe_open as safe_open_pt

GGUF = ROOT / "models/bonsai/8B/gguf/Bonsai-8B-Q1_0.gguf"
BASE_DIR = ROOT / "models/bonsai/8B/base"

# Build base shard map
base_shards = sorted(BASE_DIR.glob("*.safetensors"))
key_to_shard = {}
for s in base_shards:
    with safe_open_pt(str(s), framework="pt") as f:
        for k in f.keys():
            key_to_shard[k] = s


def load_base(name: str) -> np.ndarray:
    s = key_to_shard[name]
    with safe_open_pt(str(s), framework="pt") as f:
        t = f.get_tensor(name)
        return t.to(torch.float32).numpy()


def gguf_to_hf(gguf_name: str) -> str:
    body = gguf_name[: -len(".weight")]
    import re
    m = re.match(r"^blk\.(\d+)\.(.+)$", body)
    idx, sub = m.group(1), m.group(2)
    sub_map = {
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    return f"model.layers.{idx}.{sub_map[sub]}.weight"


reader = GGUFReader(str(GGUF), "r")


def load_gguf_q1(name: str):
    """Returns (deq_hf, hf_shape, scales_per_block_f32, signs_per_block_int8)."""
    t = next(t for t in reader.tensors if t.name == name)
    gguf_shape = list(t.shape)
    hf_shape = list(reversed(gguf_shape))
    n = int(np.prod(gguf_shape))
    raw = bytes(t.data.tobytes())
    scales, signs = parse_q1_0(raw, n)
    deq = (signs.astype(np.float32) * scales[:, None]).reshape(hf_shape)
    return deq, hf_shape, scales, signs


def sign_match(deq: np.ndarray, base: np.ndarray) -> float:
    nz = np.abs(base) > 0
    return float(np.mean(np.sign(deq[nz]) == np.sign(base[nz])))


# =================== Test 1: per-projection sign-match ===================
print("\n=== TEST 1: per-projection sign-match at L0 and L17 ===")
projs = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
labels = ["q", "k", "v", "o", "gate", "up", "down"]
results = {}
for L in [0, 17]:
    print(f"\nLayer {L}:")
    for p, lbl in zip(projs, labels):
        gn = f"blk.{L}.{p}.weight"
        deq, _, _, _ = load_gguf_q1(gn)
        base = load_base(gguf_to_hf(gn))
        s = sign_match(deq, base)
        results.setdefault(lbl, []).append(s)
        print(f"  {lbl:5s}: {s:.4f}")

# Ordering check: average across L0 and L17
print("\nMean across L0+L17 — ordering test:")
mean_by = {lbl: np.mean(vs) for lbl, vs in results.items()}
ordered = sorted(mean_by.items(), key=lambda kv: -kv[1])
for lbl, v in ordered:
    print(f"  {lbl:5s}: {v:.4f}")
print(f"v4 expected ordering: v > down > o > k > q > gate ≈ up")

# =================== Test 2: magnitude-graded sign flips ===================
print("\n=== TEST 2: magnitude-graded sign flips at L0.q_proj ===")
gn = "blk.0.attn_q.weight"
deq, _, _, _ = load_gguf_q1(gn)
base = load_base(gguf_to_hf(gn))
abs_b = np.abs(base).reshape(-1)
sign_b = np.sign(base).reshape(-1)
sign_d = np.sign(deq).reshape(-1)
# decile bins by |w|
order = np.argsort(abs_b)
nz_mask = abs_b[order] > 0
order = order[nz_mask]
n = order.size
flip_rates = []
for d in range(10):
    lo = d * n // 10
    hi = (d + 1) * n // 10
    idx = order[lo:hi]
    fr = float(np.mean(sign_b[idx] != sign_d[idx]))
    print(f"  d{d+1}  |w|∈[{abs_b[idx].min():.5f},{abs_b[idx].max():.5f}]  flip={fr:.4f}  n={idx.size}")
    flip_rates.append(fr)
print(f"\nv4 claim: d1 ≈ 0.47, d10 ≈ 0.025 (or 0.46→0.003-0.025)")
print(f"observed: d1={flip_rates[0]:.4f}, d10={flip_rates[-1]:.4f}")

# =================== Test 3: per-block scale != mean(|w_teacher|_g) ===================
print("\n=== TEST 3: per-block scale != mean(|w_teacher|_g) on 3 tensors ===")
def per_block_scale_compare(gn: str, hfn: str):
    _, hf_shape, scales_bonsai, _ = load_gguf_q1(gn)
    base = load_base(hfn)
    flat = base.reshape(-1).astype(np.float32)
    blocks = flat.reshape(-1, 128)
    mean_abs = np.mean(np.abs(blocks), axis=-1)
    # scales_bonsai is FP16-cast-back; we need to compare in FP16 storage precision
    # We'll check (a) any byte-equal blocks (FP16 representation match) and
    # (b) the median ratio.
    mean_abs_fp16 = mean_abs.astype(np.float16).astype(np.float32)
    scales_fp16 = scales_bonsai.astype(np.float16).astype(np.float32)
    # Byte-equal: same FP16 bytes
    eq_byte = float(np.mean(scales_fp16 == mean_abs_fp16))
    # Median ratio
    nz = mean_abs > 0
    ratio = scales_bonsai[nz] / mean_abs[nz]
    return eq_byte, float(np.median(ratio))


for gn in ["blk.0.attn_q.weight", "blk.17.attn_v.weight", "blk.0.ffn_down.weight"]:
    hfn = gguf_to_hf(gn)
    eqf, med = per_block_scale_compare(gn, hfn)
    print(f"  {gn}: byte-equal={eqf*100:.4f}%  median(s_bonsai/mean(|w_teacher|_g))={med:.3f}")

# =================== Test 4: per-row Pearson ===================
print("\n=== TEST 4: per-row Pearson correlation ===")
def per_row_pearson(gn: str, hfn: str):
    deq, hf_shape, scales, _ = load_gguf_q1(gn)
    # rows index out_features. blocks of 128 are along the fast (input) dim.
    out_dim, in_dim = hf_shape
    # scales shape (nblocks,); each row has in_dim/128 blocks
    bpr = in_dim // 128
    scales_2d = scales.reshape(out_dim, bpr)
    row_mean_s = scales_2d.mean(axis=1)
    base = load_base(hfn)
    row_mean_abs = np.abs(base).mean(axis=1)
    pear = float(np.corrcoef(row_mean_s, row_mean_abs)[0, 1])
    return pear

for gn in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.17.attn_q.weight", "blk.17.ffn_gate.weight"]:
    hfn = gguf_to_hf(gn)
    p = per_row_pearson(gn, hfn)
    print(f"  {gn}: Pearson(per-row mean s_g, per-row mean |w_teacher|) = {p:.4f}")
print("v4 claim: attn ~0.8, MLP gate/up ~0.42")
