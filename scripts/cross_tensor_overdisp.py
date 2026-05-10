"""Cross-tensor over-dispersion measurement.

For each (layer, projection-type), measure the per-block flip-count
over-dispersion (observed-variance / Binomial-expected-variance).
Tests whether the within-block correlation finding from 37_* is
uniform across tensors or varies by depth/type.
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
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    return f"model.layers.{idx}.{sub_map[sub]}.weight"


reader = GGUFReader(str(GGUF), "r")


def overdisp_for(gguf_name):
    """Return (flip_rate, overdispersion ratio) overall for the tensor."""
    t = next(t for t in reader.tensors if t.name == gguf_name)
    gguf_shape = list(t.shape)
    hf_shape = list(reversed(gguf_shape))
    n = int(np.prod(gguf_shape))
    raw = bytes(t.data.tobytes())
    scales, signs = parse_q1_0(raw, n)
    # signs is shape (nblocks*128,) in {-1,+1}; first reshape to blocks
    signs2d = signs.reshape(-1, 128)  # (nblocks, 128)
    base = load_base(gguf_to_hf(gguf_name)).reshape(-1, 128)
    teacher_signs = np.sign(base)
    # treat teacher zero positions as random — exclude them
    # flip = (signs2d != teacher_signs) for nz positions
    nz = teacher_signs != 0
    # per-block flip count: but blocks have variable nz — restrict to all-nz blocks
    # For simplicity: use only blocks with no teacher zeros (nearly all of them)
    full_nz_mask = nz.all(axis=1)
    sb = signs2d[full_nz_mask]
    tb = teacher_signs[full_nz_mask]
    flips = (sb != tb).sum(axis=1)  # (n_full_blocks,) flip count in {0..128}
    p = flips.mean() / 128.0
    obs_var = flips.var()
    exp_var = 128.0 * p * (1.0 - p)
    return float(p), float(obs_var / exp_var) if exp_var > 0 else float("nan"), int(full_nz_mask.sum())


probes = [
    ("attn_q", [0, 18, 35]),
    ("attn_k", [0, 18, 35]),
    ("attn_v", [0, 18, 35]),
    ("attn_output", [0, 18, 35]),
    ("ffn_gate", [0, 18, 35]),
    ("ffn_up", [0, 18, 35]),
    ("ffn_down", [0, 18, 35]),
]
print(f"{'tensor':<22}  {'flip-rate':>9}  {'overdisp':>9}  {'n_blocks':>9}")
for kind, layers in probes:
    for L in layers:
        gn = f"blk.{L}.{kind}.weight"
        try:
            p, od, nb = overdisp_for(gn)
            label = f"{kind} L{L}"
            print(f"{label:<22}  {p:>9.4f}  {od:>9.3f}  {nb:>9d}")
        except Exception as e:
            print(f"{kind} L{L}: ERR {e}")
