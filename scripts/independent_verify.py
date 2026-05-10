"""Independent verification of 4 claims about Bonsai-4B vs Qwen3-4B-base."""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/user/reversing-bonsai")
sys.path.insert(0, str(REPO / "src"))
from q1_0 import parse_q1_0, QK1_0  # noqa
from gguf import GGUFReader, GGMLQuantizationType

GGUF_PATH = REPO / "models/bonsai/4B/gguf/Bonsai-4B-Q1_0.gguf"
UNPACKED_DIR = REPO / "models/bonsai/4B/unpacked"
BASE_DIR = REPO / "models/bonsai/4B/base"


def bf16_bytes_to_f32(buf, shape):
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32).reshape(shape)


def read_safetensors_raw(path: Path, name: str):
    with open(path, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hdr_len))
        data_start = 8 + hdr_len
        if name not in hdr:
            return None
        meta = hdr[name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        off_a, off_b = meta["data_offsets"]
        fh.seek(data_start + off_a)
        buf = fh.read(off_b - off_a)
    return dtype, shape, buf


def load_from_dir(dir_path: Path, name: str):
    """Returns (raw_arr_in_native_dtype_as_f32, dtype_str). Reads BF16 raw, F16 raw."""
    for sf in sorted(dir_path.glob("*.safetensors")):
        r = read_safetensors_raw(sf, name)
        if r is None:
            continue
        dtype, shape, buf = r
        if dtype == "BF16":
            return bf16_bytes_to_f32(buf, shape), "BF16", shape
        if dtype == "F16":
            return np.frombuffer(buf, dtype=np.float16).reshape(shape).astype(np.float32), "F16", shape
        if dtype == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy(), "F32", shape
        raise ValueError(f"unknown dtype {dtype}")
    return None


def load_q1_dequant(reader: GGUFReader, gguf_name: str):
    for t in reader.tensors:
        if t.name == gguf_name:
            if t.tensor_type != GGMLQuantizationType.Q1_0:
                raise TypeError(f"{gguf_name} is {t.tensor_type.name}")
            gguf_shape = list(t.shape)
            hf_shape = list(reversed(gguf_shape))
            n_elems = int(np.prod(gguf_shape))
            raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
            scales, signs = parse_q1_0(raw, n_elems)
            flat = (signs.astype(np.float32) * scales[:, None]).reshape(-1)
            return flat.reshape(hf_shape), scales, signs
    raise KeyError(gguf_name)


# --- CLAIM 1: dequant(Bonsai-Q1_0) == Bonsai-unpacked, 3 spot tensors ---
def claim1():
    print("\n========== CLAIM 1: dequant(Q1_0) == unpacked ==========")
    reader = GGUFReader(str(GGUF_PATH), "r")
    pairs = [
        ("blk.2.attn_q.weight",  "model.layers.2.self_attn.q_proj.weight"),
        ("blk.18.ffn_down.weight","model.layers.18.mlp.down_proj.weight"),
        ("blk.34.attn_output.weight","model.layers.34.self_attn.o_proj.weight"),
    ]
    for gname, hname in pairs:
        deq, scales, signs = load_q1_dequant(reader, gname)
        loaded = load_from_dir(UNPACKED_DIR, hname)
        if loaded is None:
            print(f"  MISSING: {hname}")
            continue
        un, dtype, shape = loaded
        # Compare element-wise
        diff = deq - un
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        # Sign agreement (nonzero)
        nz = un != 0
        sign_agree = float(np.mean(np.sign(deq[nz]) == np.sign(un[nz])))

        # Compute "fp16 ULP" relative metric: ULP at value |x| in fp16 is 2^(floor(log2|x|)-10)
        # For sanity, max_abs should be 0 if storage matches exactly.
        # Cast deq -> fp16 -> back to f32 to test fp16 equality.
        deq_fp16 = deq.astype(np.float16).astype(np.float32)
        un_fp16  = un.astype(np.float16).astype(np.float32)
        max_after_fp16 = float(np.max(np.abs(deq_fp16 - un_fp16)))

        # ULP threshold: max ulp at largest magnitude in tensor
        max_mag = float(np.max(np.abs(un)))
        # ULP of fp16 at magnitude m: ~ m * 2^-10
        ulp_at_max = max_mag * 2**-10

        print(f"  {gname} -> {hname}  shape={shape}  unpacked_dtype={dtype}")
        print(f"    max|deq - un|       = {max_abs:.6g}")
        print(f"    mean|deq - un|      = {mean_abs:.6g}")
        print(f"    max|deq_fp16 - un|  = {max_after_fp16:.6g}")
        print(f"    sign_agree          = {sign_agree*100:.4f}%")
        print(f"    1-ULP at max(|w|)~  = {ulp_at_max:.6g}  (max(|w|)={max_mag:.4g})")


# --- CLAIM 2: greedy best-row-perm cosine == identity-row cosine within 1e-3 ---
def cosine_per_row(a, b):
    an = np.linalg.norm(a, axis=-1) + 1e-12
    bn = np.linalg.norm(b, axis=-1) + 1e-12
    return np.einsum("ij,ij->i", a, b) / (an * bn)


def best_row_perm_greedy(bonsai, base, max_rows=4096, seed=0):
    rows = bonsai.shape[0]
    if rows > max_rows:
        idx = np.random.default_rng(seed).choice(rows, size=max_rows, replace=False)
        bonsai_s = bonsai[idx]; base_s = base[idx]
    else:
        bonsai_s = bonsai; base_s = base
    a = bonsai_s / (np.linalg.norm(bonsai_s, axis=-1, keepdims=True) + 1e-12)
    b = base_s   / (np.linalg.norm(base_s,   axis=-1, keepdims=True) + 1e-12)
    sim = b @ a.T
    n = sim.shape[0]
    used = np.zeros(n, dtype=bool)
    perm = np.full(n, -1, dtype=np.int64)
    order = np.argsort(-sim.max(axis=1))
    diag_id = float(np.einsum("ij,ij->i", a, b).mean())
    for i in order:
        cands = np.argsort(-sim[i])
        for c in cands:
            if not used[c]:
                perm[i] = c; used[c] = True; break
    matched = float(sim[np.arange(n), perm].mean())
    return diag_id, matched, perm


def claim2():
    print("\n========== CLAIM 2: best-row-perm vs identity row-cosine ==========")
    name = "model.layers.10.self_attn.q_proj.weight"
    bonsai_loaded = load_from_dir(UNPACKED_DIR, name)
    base_loaded   = load_from_dir(BASE_DIR, name)
    if bonsai_loaded is None or base_loaded is None:
        print("  MISSING tensors"); return
    bw, _, bshape = bonsai_loaded
    qw, _, qshape = base_loaded
    print(f"  tensor: {name}  shapes bonsai={bshape} base={qshape}")
    if bw.shape != qw.shape:
        print("  shape mismatch!"); return
    cos_id = float(cosine_per_row(bw, qw).mean())
    diag_id, best, perm = best_row_perm_greedy(bw, qw, max_rows=4096)
    print(f"    identity row cosine (full)         = {cos_id:.6f}")
    print(f"    identity row cosine (sample 4096)  = {diag_id:.6f}")
    print(f"    best-greedy-perm  row cosine       = {best:.6f}")
    print(f"    delta (best - identity)            = {best - diag_id:.6e}")
    print(f"    perm is identity?                  = {bool((perm == np.arange(len(perm))).all())}")


# --- CLAIM 3: naive PTQ Q1_0_g128 on Qwen3-base -> row cos ~ 0.798 ---
def claim3():
    print("\n========== CLAIM 3: PTQ baseline ~0.798 ==========")
    name = "model.layers.10.self_attn.q_proj.weight"
    base_loaded = load_from_dir(BASE_DIR, name)
    if base_loaded is None:
        print("  MISSING"); return
    qw, dtype, shape = base_loaded
    print(f"  base tensor {name} dtype={dtype} shape={shape}")
    flat = qw.reshape(-1)
    assert flat.size % QK1_0 == 0, f"size {flat.size} not multiple of {QK1_0}"
    blocks = flat.reshape(-1, QK1_0)
    scales = np.mean(np.abs(blocks), axis=-1)              # mean(|w|) per group
    signs  = np.where(blocks >= 0.0, 1.0, -1.0).astype(np.float32)
    quant  = (signs * scales[:, None]).reshape(qw.shape)
    cos_id = float(cosine_per_row(quant, qw).mean())
    print(f"    PTQ-quant row cosine vs base = {cos_id:.6f}")
    print(f"    sqrt(2/pi) prediction        = {np.sqrt(2/np.pi):.6f}")
    print(f"    delta                        = {cos_id - np.sqrt(2/np.pi):.6f}")


# --- CLAIM 4: norms differ from base by more than BF16 ULP ---
def claim4():
    print("\n========== CLAIM 4: norms re-trained ==========")
    name = "model.layers.0.self_attn.q_norm.weight"
    bonsai_loaded = load_from_dir(UNPACKED_DIR, name)
    base_loaded   = load_from_dir(BASE_DIR, name)
    if bonsai_loaded is None or base_loaded is None:
        print(f"  MISSING ({name})"); return
    bn_arr, bn_dtype, bn_shape = bonsai_loaded
    bs_arr, bs_dtype, bs_shape = base_loaded
    print(f"  tensor: {name}")
    print(f"    bonsai dtype={bn_dtype} shape={bn_shape}  base dtype={bs_dtype} shape={bs_shape}")
    # Bonsai is FP16 in storage, base is BF16. Cast base BF16 -> FP16 round-trip
    # (already in float32; downcast to fp16 to mimic FP16 storage).
    bs_fp16 = bs_arr.astype(np.float16).astype(np.float32)
    bn_fp16 = bn_arr.astype(np.float16).astype(np.float32)
    diff = bn_fp16 - bs_fp16
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    # BF16 ULP at value magnitude m: m * 2^-7
    max_mag = float(np.max(np.abs(bs_arr)))
    bf16_ulp_at_max = max_mag * 2**-7
    fp16_ulp_at_max = max_mag * 2**-10
    print(f"    max|bonsai - base|         = {max_abs:.6g}")
    print(f"    mean|bonsai - base|        = {mean_abs:.6g}")
    print(f"    BF16 ULP at max(|w|={max_mag:.4g}) = {bf16_ulp_at_max:.6g}")
    print(f"    FP16 ULP at max(|w|)              = {fp16_ulp_at_max:.6g}")
    # Also report some samples
    print(f"    first 8 base values  = {bs_arr.flatten()[:8]}")
    print(f"    first 8 bonsai vals  = {bn_arr.flatten()[:8]}")
    # Relative diff
    rel = float(np.mean(np.abs(diff) / (np.abs(bs_arr) + 1e-9)))
    print(f"    mean relative diff   = {rel:.6g}")


if __name__ == "__main__":
    claim1()
    claim2()
    claim3()
    claim4()
