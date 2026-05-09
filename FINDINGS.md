# Reversing Bonsai — current notes

> Status: pre-empirical. HuggingFace egress is blocked on the Claude Code web
> sandbox (`x-deny-reason: host_not_allowed`), so the real GGUFs are unavailable
> in this environment. The GitHub Actions workflow at
> `.github/workflows/analyze-bonsai.yml` runs the same analysis on a hosted
> runner where HF is reachable.

## What the paper(s) say

- **Q1_0_g128 format.** One sign bit per weight, one shared FP16 scale per
  group of 128 contiguous weights. Reconstruction: `w_i = s_g · (2 b_i − 1)`
  with `b_i ∈ {0,1}`. Effective storage 1 + 16/128 = **1.125 bits/weight**.
- **Applied to**: embeddings, attention projections, MLP projections, LM head.
  Norms (RMSNorm) and small scale tensors stay in higher precision.
- **Base architectures unchanged**: Qwen3-{1.7B, 4B, 8B}.
- The methodology that makes 1-bit quality possible is described as
  "proprietary Caltech IP" — the paper does not disclose it. (This is exactly
  what we're trying to reverse.)
- A **Ternary-Bonsai** family exists in `{−1, 0, +1}` with the same group-128
  scheme, packed in `Q2_0` (≈1.71 bpw on disk).

## What llama.cpp's source says (verified against `ggml/src/`)

### Block layout (`ggml-common.h:177`)
```c
#define QK1_0 128
typedef struct {
    ggml_half d;             // FP16 scale
    uint8_t   qs[QK1_0 / 8]; // 16 bytes -> 128 sign bits
} block_q1_0;                // sizeof == 18
```

### Quantize ref (`ggml-quants.c:36`)
```c
const float d = sum_abs / qk;     // scale = MEAN of |x| in the group
y[i].d = GGML_FP32_TO_FP16(d);
// bit at element j -> qs[j/8] bit (j%8); set iff x[j] >= 0
```

Two consequences worth highlighting:

1. The scale is the **mean magnitude**, not max-abs. This is unusual — most
   quant formats use max-abs. Mean magnitude minimizes sum-abs reconstruction
   error rather than max-abs, which matches the symmetric ±d codebook (any
   value other than mean(|x|) shifts every entry's residual the same way).
2. The signs are written in **little-endian within byte** (LSB = bit 0). My
   pure-Python codec at `src/q1_0.py` matches this and round-trips byte-for-byte.

### Bit packing
Element index `j ∈ [0, 128)` maps to `qs[j // 8]`, bit `j % 8`. Bits within a
byte are LSB-first. `np.unpackbits(qs, bitorder="little")` reproduces this.

## The user's two questions, recast

### Q1. "Can Q1_0 be unpacked into the same FP representation as base Qwen3?"

For a *lossless* identity, every original Qwen3 weight inside a 128-group must
already lie on `{±d_g}` with `d_g = mean(|x|)` of the group. That's a hard
constraint Qwen3 does NOT naturally satisfy. So either:

- (a) Bonsai is a **QAT / distilled** Qwen3-shaped network whose weights have
  been trained to live on a binary lattice. The `Bonsai-{size}-unpacked` HF
  repo (separate from the GGUF repo) exists precisely to ship those FP16
  values for backends that can't yet do 1-bit kernels — the values are still
  on the binary lattice but stored as FP16. **Strong hypothesis.**
- (b) The "unpacked" model is the dequantized Q1_0 (i.e. literally the output
  of `dequantize_row_q1_0`). In this case `Bonsai-unpacked == dequant(Bonsai-Q1_0)`
  and is *not* the original Qwen3. This is also consistent with the user's
  phrasing "can be unpacked into the same floating representation".

(a) and (b) are testable in the workflow:

- If `Bonsai-unpacked` and `dequant(Bonsai-Q1_0)` agree to fp16 precision
  *element-wise*, the unpacked file is the identical binary lattice in FP16
  storage.
- If `Bonsai-unpacked` row-by-row is cosine-similar to `Qwen3-base` after some
  permutation, then Bonsai is "Qwen3 reordered + sign-quantized + retrained".

### Q2. "Are weights sorted within each 128 group? Could the graph have been reordered?"

A linear layer `y = W x + b` is invariant under joint permutations of:
- `W`'s rows + the next layer's columns (output-channel permutation),
- `W`'s columns + the previous layer's rows (input-channel permutation),
- and within an MLP, the FFN intermediate dim is fully permutable.

So Bonsai *could* reorder Qwen3's channels so that each 128-block becomes
"easier" to express with a single magnitude (e.g. by clustering similar
magnitudes together) without changing the function. Hidden constraints: in
GQA (Qwen3 uses 32 query / 8 KV heads), the head structure ties together
groups of 128 channels (head_dim=128 in Qwen3-1.7B). RoPE rotates *pairs*
inside each head, so even pair ordering matters. So the legal permutations
are:

- Per-head reordering of output channels in `attn_q`, `attn_k`, `attn_v`
  is constrained: must respect the head boundary (and RoPE pair structure
  for q/k).
- FFN intermediate dim (`ffn_gate`/`ffn_up`/`ffn_down`) is freely permutable.
- Embedding rows ↔ vocab order: only relabels tokens; immovable.
- LM head columns ↔ residual stream: must match attn-output / ffn-down.

**Empirical tests in the workflow:**

- `analyze_q1_0.py` measures, per Q1_0 tensor:
  - mean number of sign transitions per block (random ≈ 64; sorted ≤ 1);
  - fraction of blocks with `≤1` sign transition (a "sorted" block is
    `[−,−,…,−,+,+,…,+]` or all-same);
  - lag-1 sign autocorrelation;
  - fraction of adjacent block-scale pairs that are non-decreasing
    (would be ~50% for random ordering, ~100% if scales were globally
    sorted).
- `compare_unpacked_vs_qwen3.py` measures:
  - row-wise cosine of `Bonsai-unpacked` vs `Qwen3-base` (identity);
  - greedy best row permutation cosine (lower-bound on how well any
    permutation fits);
  - whether the lattice constraint holds (per-128-group: how many distinct
    `|w|` values are present — "1" means binary lattice).

Both scripts are dimension-aware: GGUF stores tensors with the fastest dim
first, so a `(out, in)` weight matrix in safetensors becomes `[in, out]` in
GGUF metadata. Q1_0 blocks span the **fastest dim** (`in` for a column-major
weight, which is rows when read back as numpy). That's the relevant axis for
"is this block sorted in input-channel order".

## What this means for the scientific question

- **If signs within blocks are random** (transitions ≈ 64, ac1 ≈ 0): Bonsai
  did not reorder for sign-clustering. The "magic" is in *which* sign+scale
  combination each block carries, not in the ordering.
- **If `|Bonsai-unpacked|` is constant per 128-block** (`distinct_magnitudes
  per group == 1`): the unpacked model lives on the binary lattice as
  predicted; it's just an FP16 *storage* of a 1-bit *value space*.
- **If `Bonsai-unpacked` row ↔ Qwen3-base row identity cosine is low but
  best-permutation cosine is high**: Bonsai applied a learned permutation to
  Qwen3 channels before quantization (consistent with QAT + knowledge
  distillation that frees up channel reorderings).
- **If both cosines are low**: Bonsai is a substantially different model
  that retrained from scratch with the binary constraint, only inheriting
  the architecture (and possibly tokenizer + first-layer init) from Qwen3.

The workflow produces three reports in the artifact bundle that together
distinguish these cases.

## Open questions only the user can answer / artifacts can resolve

1. Does `prism-ml/Bonsai-{size}-unpacked` actually exist? My WebSearch found
   it; the workflow tolerates "missing" if not.
2. The base used for distillation — is it instruct or base Qwen3? The paper
   compares against Qwen3-{size} instruct on benchmarks; the GGUF metadata
   `general.base_model` field (if set) will tell us which checkpoint.
3. Is the per-tensor scale `d` exactly `mean(|x|)`, or did Bonsai overrride
   to something learned? Easy to check: load `Bonsai-unpacked`, compute
   `mean(|x|)` per group, and compare to the FP16 scale stored in the
   Q1_0 file at the same group.

## Files in this branch

- `src/q1_0.py` — pure-Python Q1_0 codec mirroring `ggml-quants.c`.
- `src/gguf_inspect.py` — dump GGUF metadata + tensor inventory.
- `src/analyze_q1_0.py` — sign-pattern, run-length, and scale-ordering stats
  across every Q1_0 tensor.
- `src/compare_unpacked_vs_qwen3.py` — element-wise + permutation-tolerant
  comparison of two FP16 checkpoints.
- `.github/workflows/analyze-bonsai.yml` — manual-trigger workflow that runs
  the above on a hosted runner with HF egress.

To run: open the repo on github.com, Actions tab → "Analyze Bonsai
Quantization" → "Run workflow", pick size=1.7B (fastest), keep both
`include_unpacked` and `include_base` enabled. The artifacts upload contains
the full text reports.
