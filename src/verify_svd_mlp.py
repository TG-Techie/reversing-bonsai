# Independent verification of SVD low-rank claim and MLP-SGD concentration claim.
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_0 import parse_q1_0, QK1_0
from compare_q1_dequant_vs_unpacked import gguf_to_hf_candidates

GGUF = "models/bonsai/8B/gguf/Bonsai-8B-Q1_0.gguf"
SHARD = "models/bonsai/8B/base/model-00003-of-00005.safetensors"
TARGETS = {
    "model.layers.18.self_attn.q_proj.weight",
    "model.layers.18.mlp.gate_proj.weight",
}


def load_bonsai_tensors():
    """Return dict[hf_name] -> (W_bonsai_fp32, scales_per_block, signs_per_block)."""
    reader = GGUFReader(GGUF, "r")
    out = {}
    for t in reader.tensors:
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            continue
        cands = gguf_to_hf_candidates(t.name)
        if not cands or cands[0] not in TARGETS:
            continue
        hf = cands[0]
        gguf_shape = list(t.shape)
        hf_shape = list(reversed(gguf_shape))
        raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
        scales, signs = parse_q1_0(raw, int(np.prod(gguf_shape)))
        # element values
        W = (signs.astype(np.float32) * scales[:, None]).reshape(hf_shape)
        out[hf] = (W, scales, signs, hf_shape)
        print(f"  loaded {hf} shape={hf_shape} fast-dim={hf_shape[-1]}")
    return out


def load_teacher_tensors():
    # Qwen3 base is bf16 — safetensors numpy can't read bf16 directly.
    # Use torch to load and convert.
    import torch
    from safetensors.torch import safe_open as t_safe_open
    out = {}
    with t_safe_open(SHARD, framework="pt") as f:
        for k in TARGETS:
            t = f.get_tensor(k).to(torch.float32).numpy()
            out[k] = t
    return out


def svd_rank_explained(delta, ranks):
    """Return dict rank -> fraction of squared-Frobenius explained by top-r singular values."""
    # Use SVD without computing U,V to save memory
    s = np.linalg.svd(delta.astype(np.float32), compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    out = {}
    for r in ranks:
        out[r] = float(s2[:r].sum() / total)
    return out


def mlp_sgd_match(W_bonsai_fp32, scales_bonsai, hf_shape, W_teacher, sigma_mult=1.25,
                  rel_tol=0.10, seed=0):
    """Replicate the MLP-SGD test. Returns the overall fraction of blocks where
    Q1_0(teacher + Gaussian noise) scale matches Bonsai's deployed scale within rel_tol."""
    rng = np.random.default_rng(seed)
    teacher_flat = W_teacher.reshape(-1).astype(np.float32)
    n = teacher_flat.size
    teacher_mean_abs = np.mean(np.abs(teacher_flat))
    sigma = sigma_mult * teacher_mean_abs
    # add Gaussian noise (LoRA stand-in)
    sim = teacher_flat + rng.normal(0.0, sigma, size=n).astype(np.float32)
    # Q1_0 scale = mean(|x|) over each block of QK1_0
    nb = n // QK1_0
    sim_blocks = sim[: nb * QK1_0].reshape(nb, QK1_0)
    s_lora_sim = np.mean(np.abs(sim_blocks), axis=1)
    s_bonsai = scales_bonsai.astype(np.float32)
    assert s_bonsai.size == nb, (s_bonsai.size, nb)
    rel = np.abs(s_lora_sim - s_bonsai) / np.maximum(s_bonsai, 1e-12)
    overall = float(np.mean(rel <= rel_tol))
    # per-row stats
    rows = hf_shape[0]
    cols = hf_shape[1]
    blocks_per_row = cols // QK1_0
    assert blocks_per_row * rows == nb
    matched = (rel <= rel_tol).reshape(rows, blocks_per_row)
    per_row_frac = matched.mean(axis=1)
    rows_gt60 = float(np.mean(per_row_frac > 0.60))
    rows_lt20 = float(np.mean(per_row_frac < 0.20))
    return overall, rows_gt60, rows_lt20, rel


def main():
    print("Loading Bonsai (Q1_0 dequantized)...")
    bonsai = load_bonsai_tensors()
    print("Loading teacher (Qwen3 base)...")
    teacher = load_teacher_tensors()
    for k in TARGETS:
        assert k in bonsai, f"Bonsai missing {k}"
        assert k in teacher, f"Teacher missing {k}"
        Wb, scales, signs, hf_shape = bonsai[k]
        Wt = teacher[k]
        print(f"\n=== {k} ===")
        print(f"  shape bonsai={Wb.shape} teacher={Wt.shape}")
        assert Wb.shape == Wt.shape

    # Test 1: SVD on q_proj L18
    k = "model.layers.18.self_attn.q_proj.weight"
    Wb, scales, signs, hf_shape = bonsai[k]
    Wt = teacher[k]
    delta = (Wb - Wt).astype(np.float32)
    fro_norm = float(np.linalg.norm(delta))
    print(f"\n[SVD test] {k}")
    print(f"  delta Frobenius = {fro_norm:.3f}")
    print(f"  delta mean|.|   = {float(np.mean(np.abs(delta))):.4f}")
    explained = svd_rank_explained(delta, [1, 16, 64, 128, 256, 1024])
    for r, frac in explained.items():
        print(f"  rank {r:>4}: {100*frac:6.2f}%")

    # Test 2: MLP-SGD on gate_proj L18
    k = "model.layers.18.mlp.gate_proj.weight"
    Wb, scales, signs, hf_shape = bonsai[k]
    Wt = teacher[k]
    print(f"\n[MLP-SGD test] {k}")
    print(f"  shape={hf_shape} teacher mean|w|={float(np.mean(np.abs(Wt))):.5f}")
    print(f"  bonsai scale stats: min={scales.min():.4f} median={np.median(scales):.4f} max={scales.max():.4f}")
    overall, rgt60, rlt20, rel = mlp_sgd_match(Wb, scales, hf_shape, Wt,
                                                sigma_mult=1.25, rel_tol=0.10, seed=0)
    print(f"  overall match (sigma=1.25*teacher_mean, tol=10%) = {overall:.4f}")
    print(f"  rows >60% matched: {100*rgt60:.1f}%   rows <20% matched: {100*rlt20:.1f}%")
    # robustness: try a couple of seeds and a different sigma
    for seed in [1, 2]:
        o, _, _, _ = mlp_sgd_match(Wb, scales, hf_shape, Wt, 1.25, 0.10, seed=seed)
        print(f"  [seed={seed}] overall match = {o:.4f}")
    # alt sigma: sigma=teacher_mean (1.0x)
    for mult in [0.5, 1.0, 2.0]:
        o, _, _, _ = mlp_sgd_match(Wb, scales, hf_shape, Wt, mult, 0.10, seed=0)
        print(f"  [sigma_mult={mult}] overall match = {o:.4f}")

    # Sanity: if we feed pure teacher (no noise) what fraction matches Bonsai scale within 10%?
    Wt_flat = Wt.reshape(-1)
    nb = Wt_flat.size // QK1_0
    s_teacher = np.mean(np.abs(Wt_flat[:nb*QK1_0].reshape(nb, QK1_0)), axis=1)
    rel_t = np.abs(s_teacher - scales) / np.maximum(scales, 1e-12)
    print(f"  [no-noise sanity] fraction of teacher-blocks within 10% of Bonsai scale = {float(np.mean(rel_t<=0.10)):.4f}")
    # what does Gaussian-only (teacher set to zero) produce?
    Wzero = np.zeros_like(Wt)
    o_zero, _, _, _ = mlp_sgd_match(Wb, scales, hf_shape, Wzero, 1.25, 0.10, seed=0)
    print(f"  [teacher=0, only Gaussian] overall match = {o_zero:.4f}")


if __name__ == "__main__":
    main()
