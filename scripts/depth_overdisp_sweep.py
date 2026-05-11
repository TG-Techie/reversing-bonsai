"""Full 36-layer over-dispersion sweep for q, mlp.gate, mlp.up at 8B.

Refines 38_* (3 depths sampled) to all 36 layers. Sees the depth
shape clearly: monotonic, plateau, U-shaped, or jump?

Outputs CSV-style table for easy plotting / analysis.
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
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
    }
    return f"model.layers.{idx}.{sub_map[sub]}.weight"


reader = GGUFReader(str(GGUF), "r")
# Pre-index Q1_0 tensors by name for O(1) lookup
q1_idx = {t.name: t for t in reader.tensors}


def overdisp_for(gguf_name):
    t = q1_idx[gguf_name]
    gguf_shape = list(t.shape)
    n = int(np.prod(gguf_shape))
    raw = bytes(t.data.tobytes())
    _, signs = parse_q1_0(raw, n)
    signs2d = signs.reshape(-1, 128)
    base = load_base(gguf_to_hf(gguf_name)).reshape(-1, 128)
    teacher_signs = np.sign(base)
    full_nz_mask = (teacher_signs != 0).all(axis=1)
    sb = signs2d[full_nz_mask]
    tb = teacher_signs[full_nz_mask]
    flips = (sb != tb).sum(axis=1)
    p = flips.mean() / 128.0
    obs_var = flips.var()
    exp_var = 128.0 * p * (1.0 - p)
    return float(p), float(obs_var / exp_var) if exp_var > 0 else float("nan")


kinds = ["attn_q", "attn_v", "ffn_gate", "ffn_up", "ffn_down"]
print(f"{'L':>3} | {'q_p':>6} {'q_od':>6} | {'v_p':>6} {'v_od':>6} | {'gp_p':>6} {'gp_od':>6} | {'up_p':>6} {'up_od':>6} | {'dn_p':>6} {'dn_od':>6}")
for L in range(36):
    row = [f"{L:>3}"]
    for kind in kinds:
        gn = f"blk.{L}.{kind}.weight"
        try:
            p, od = overdisp_for(gn)
            row.append(f"{p:6.4f} {od:6.3f}")
        except Exception as e:
            row.append(f"  err   err")
    print(" | ".join([row[0]] + row[1:]))
