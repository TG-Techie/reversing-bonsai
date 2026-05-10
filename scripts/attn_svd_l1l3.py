"""SVD on (W_bonsai - W_teacher) at attention q at L0, L1, L2, L3, L18, L35.

Confirms whether the L1-3 rank-concentration spike at MLP also
appears at attention (same depth) or is MLP-specific.

q at 8B is (4096, 4096) — quick SVD.
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

base_shards = sorted(BASE_DIR.glob("*.safetensors"))
key_to_shard = {}
for s in base_shards:
    with safe_open_pt(str(s), framework="pt") as f:
        for k in f.keys():
            key_to_shard[k] = s


def load_base(name):
    s = key_to_shard[name]
    with safe_open_pt(str(s), framework="pt") as f:
        return f.get_tensor(name).to(torch.float32).numpy()


def gguf_to_hf(gguf_name):
    body = gguf_name[: -len(".weight")]
    import re
    m = re.match(r"^blk\.(\d+)\.(.+)$", body)
    idx, sub = m.group(1), m.group(2)
    sub_map = {
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
    }
    return f"model.layers.{idx}.{sub_map[sub]}.weight"


reader = GGUFReader(str(GGUF), "r")
q1_idx = {t.name: t for t in reader.tensors}


def load_q1_to_hf(gguf_name):
    t = q1_idx[gguf_name]
    gguf_shape = list(t.shape)
    hf_shape = list(reversed(gguf_shape))
    n = int(np.prod(gguf_shape))
    raw = bytes(t.data.tobytes())
    scales, signs = parse_q1_0(raw, n)
    deq = (signs.astype(np.float32) * scales[:, None]).reshape(hf_shape)
    return deq


def rank_concentration(delta, ranks=(16, 64, 128, 256, 512)):
    s = np.linalg.svd(delta, compute_uv=False)
    ssq = s ** 2
    total = ssq.sum()
    out = {}
    for r in ranks:
        out[r] = float(ssq[:r].sum() / total) if total > 0 else 0.0
    return out, float(total)


probes = [("attn_q", L) for L in [0, 1, 2, 3, 18, 35]] + [("attn_v", L) for L in [0, 1, 2, 3, 18, 35]]
print(f"{'tensor':<22}  {'rank-16':>9}  {'rank-64':>9}  {'rank-128':>9}  {'rank-256':>9}  {'rank-512':>9}  {'||delta||F':>12}")
for kind, L in probes:
    gn = f"blk.{L}.{kind}.weight"
    deq = load_q1_to_hf(gn)
    base = load_base(gguf_to_hf(gn))
    delta = (deq - base).astype(np.float32)
    conc, total = rank_concentration(delta)
    label = f"{kind} L{L}"
    print(f"{label:<22}  {conc[16]*100:>8.2f}%  {conc[64]*100:>8.2f}%  {conc[128]*100:>8.2f}%  {conc[256]*100:>8.2f}%  {conc[512]*100:>8.2f}%  {total**0.5:>12.3f}")
