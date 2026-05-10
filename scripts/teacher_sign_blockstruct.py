"""Confound check: are teacher per-block signs block-correlated?

If teacher signs within 128-blocks are themselves block-correlated
(e.g., from low-rank teacher structure), the apparent over-dispersion
in Bonsai-vs-teacher flip counts could be inherited rather than
introduced by the technique.

Per-block teacher-sign-balance test:
- For each 128-block, count positive signs (range 0..128).
- Under per-element-iid teacher signs, this is Binomial(128, p_teacher).
- Over-dispersion ratio compares observed variance to Binomial.

Also check: random shuffle of teacher signs across the tensor (preserves
marginal sign rate, breaks any block structure) and re-measure.
"""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import safe_open as safe_open_pt

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


def teacher_block_sign_struct(name, label):
    base = load_base(name)
    flat = base.reshape(-1, 128)
    teacher_sign = (np.sign(flat) > 0).astype(np.int32)  # 1 for positive
    pos_per_block = teacher_sign.sum(axis=1)  # (nblocks,) in {0..128}
    p = pos_per_block.mean() / 128.0
    obs_var = pos_per_block.var()
    exp_var = 128.0 * p * (1.0 - p)
    od_obs = float(obs_var / exp_var) if exp_var > 0 else float("nan")

    # Shuffled control: randomly permute all teacher signs
    rng = np.random.default_rng(0)
    flat_signs = teacher_sign.reshape(-1).copy()
    rng.shuffle(flat_signs)
    flat_signs = flat_signs.reshape(-1, 128)
    pos_shuf = flat_signs.sum(axis=1)
    p2 = pos_shuf.mean() / 128.0
    obs_var2 = pos_shuf.var()
    exp_var2 = 128.0 * p2 * (1.0 - p2)
    od_shuf = float(obs_var2 / exp_var2) if exp_var2 > 0 else float("nan")
    print(f"  {label:<28} p_pos={p:.4f}  obs_overdisp={od_obs:.3f}  shuf_overdisp={od_shuf:.3f}  excess={od_obs-od_shuf:+.3f}")


print(f"Per-block teacher-sign balance over-dispersion vs uniform-shuffle control")
print(f"  (1.0 = i.i.d.; >1.0 = block-correlated teacher signs)\n")
for L in [0, 18, 35]:
    for kind, hf in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ]:
        name = f"model.layers.{L}.{hf}.weight"
        teacher_block_sign_struct(name, f"L{L} {kind}")
    print()
