# Reversing Bonsai — empirical findings (1.7B)

> Status: empirical. Ran on the
> [`models-bonsai-1.7B-r5`](https://github.com/TG-Techie/reversing-bonsai/releases/tag/models-bonsai-1.7B-r5)
> trio (Q1_0 GGUF, FP16 "unpacked" safetensors, BF16 Qwen3-1.7B base).
> Reproduce with `scripts/fetch_models_from_release.sh` + the commands at the
> end of this doc. Raw reports live under `reports/local/`.

## Headline answers to the three hypotheses

| # | Hypothesis | Verdict | Evidence |
| - | - | - | - |
| H1 | `dequant(Bonsai-Q1_0)` ≡ `Bonsai-unpacked` (FP16) elementwise | **✅ Confirmed** | 197/197 Q1_0 tensors. Worst max-abs diff = **9.77e-4** (1 FP16 ULP at the local scale). Sign agreement = 100.0000% on every tensor. Per-group scale `d` in Q1_0 reproduces `mean(|x|)` of the unpacked group to ≤ 3e-5. **99.9947%** of all 13.4M groups have exactly one distinct \|w\|. |
| H2 | `Bonsai-unpacked` is `Qwen3-1.7B-base` after channel permutation | **❌ Rejected** | Across early/mid/late blocks (0, 13, 27): identity-row cosine = 0.43–0.60. Greedy best-row-permutation cosine **= identity cosine** for every layer; the search literally cannot improve on identity. No reorder. |
| H3 | Signs within each 128-block are sorted/clustered | **❌ Rejected** | Mean sign transitions per block = **63.50** (random binomial expects 63.5). 0.00% of blocks have ≤ 1 transition. Lag-1 sign autocorrelation = 0.0001. Adjacent-block scales non-decreasing 50.94% of the time. Indistinguishable from random. |

## What that combination implies

The "1-bit Bonsai" magic is not in the layout. It's QAT.

1. The deployed Q1_0 GGUF and the FP16 "unpacked" file are the *same artifact*
   in two containers. The unpacked file is exactly `dequantize_row_q1_0`'s
   output, FP16-cast. There is no second high-precision representation.
2. The weights live on a strict binary lattice `±s_g` with `s_g = mean(|x_g|)`.
3. The signs were *learned*: Bonsai-unpacked agrees with Qwen3-base on roughly
   65–75% of signs (cosine 0.43–0.60 over the row at constant magnitude
   implies that, given that pure Gaussian sign-quantization yields
   `cos = sqrt(2/π) ≈ 0.80` and any sign disagreement only lowers it).
4. There is no row, column, or channel permutation between Bonsai and base —
   identical channel ordering, head boundary, and head pair structure.
5. There is no sign sortedness or scale sortedness inside a block. The
   per-block sign pattern is statistically random.

So Bonsai-1.7B is best described as: *Qwen3-1.7B with the matrix-heavy
weights replaced by signs trained on the binary lattice* `{±s_g}`, FFN
intermediate / head ordering preserved, RMSNorms and small per-head q/k norms
left in higher precision. The "proprietary Caltech IP" the paper alludes to
is the QAT recipe, not the weight format.

## Numbers in detail

### H1 — `dequant(Q1) == unpacked` (197 tensors)

```
worst max|deq - unpacked|:    9.766e-04   (blk.17.ffn_up.weight)
mean rel-rmse vs |unpacked|:  ~1e-5  to 2e-5  per tensor
sign agreement on nonzeros:   100.0000% on every tensor
binary-lattice frac (global): 0.999947  (13,436,036 / 13,436,752 groups)
mean per-tensor scale-mean-diff: 5.7e-10
27 / 197 tensors are bit-identical (max diff = 0)
```

Tensors that hit max-diff = 0 are the ones where the FP16 round-trip happens
to land exactly — many `attn_v` and a few `attn_k` weights.

### H2 — channel permutation against Qwen3-1.7B

Per-block summary (representative samples). `cos_id` is identity row cosine,
`cos_perm` is greedy best-row-permutation cosine.

```
                         lattice frac (binary)
layer.tensor              Bonsai   base    cos_id   cos_perm
layers.0.mlp.down         0.9999   0.0000  0.5815   0.5815
layers.0.mlp.gate         0.9999   0.0000  0.4784   0.4784
layers.0.mlp.up           0.9999   0.0000  0.4662   0.4660
layers.0.self_attn.q      1.0000   0.0000  0.4638   0.4635
layers.0.self_attn.k      0.9999   0.0000  0.4807   0.4807
layers.0.self_attn.o      1.0000   0.0000  0.6006   0.6006
layers.13.mlp.up          0.9999   0.0000  0.4908   0.4910
layers.13.self_attn.k     1.0000   0.0000  0.5042   0.5042
layers.27.mlp.gate        0.9999   0.0000  0.4428   0.4421
layers.27.self_attn.k     1.0000   0.0000  0.3460   0.3456
layers.27.self_attn.q     0.9999   0.0000  0.4291   0.4285
```

Two stable patterns:
- `lattice_bonsai.binary_frac` is always 1.0 or 0.9999 (binary lattice
  confirmed independently of H1 — the unpacked file *also* lives on `±s_g`).
- `cos_perm` is within 1e-3 of `cos_id` for every layer. Channel permutation
  doesn't help.

`embed_tokens` shapes don't match: Bonsai-1.7B has `(151669, 2048)`, base has
`(151936, 2048)`. Bonsai trimmed 267 vocab entries (probably reserved special
tokens). Comparison skipped for the embedding row.

### H3 — sign / scale layout in Q1_0

```
                         mean        min         max         (random)
transitions / block:     63.50       63.37       63.61       (~63.5)
frac all-same-sign:      0.0000      0.0000      0.0000      (0)
frac <=1 transition:     0.0000      0.0000      0.0000      (0)
ac_lag1 over signs:      +0.0001     -0.0018     +0.0020     (0)
pos count / block:       64.00       63.81       64.28       (64)
frac scale-pairs nondec: 50.94%      46.92%      55.73%      (50%)
distinct fp16 scales:    ~150–350 per tensor (out of ~16k–98k blocks)
```

Per-tensor scales are tightly clustered (μ ≈ 0.06, σ ≈ 0.01) but those are
the values the trainer landed on, not anything globally sorted.

## Why this is consistent with the paper

The paper claims a 9-point benchmark gap (Qwen3-8B 79.3 → Bonsai-8B 70.5) at
1/14× the storage, and attributes the breakthrough to "proprietary Caltech
IP." The empirical picture matches exactly that:

- Pure post-training sign-quant of Qwen3 would land at cos ≈ 0.80 row-wise
  and produce a model that's fluent but materially less reliable on
  multi-step tasks — which is precisely the failure mode the paper says
  prior 1-bit work suffered. Bonsai is at cos ≈ 0.50, i.e. signs were
  re-learned, not just thresholded.
- The kernel work in the paper's Appendix A (custom CUDA / Metal / OpenCL
  paths for `Q1_0_g128`) is necessary because no standard runtime
  natively dequantizes the binary lattice. That work is real, public, and in
  the PrismML llama.cpp / MLX forks.
- The training recipe is the only undisclosed piece. Reverse-engineering it
  would require either a calibration corpus + their loss schedule, or
  white-box gradient observation, neither of which we have.

## Open questions

1. **Sign agreement vs Qwen3 per layer.** We have row cosine; the next step
   is a direct sign-disagreement count vs Qwen3 to localize where Bonsai
   diverged most (intuition: later layers, since cos drops from ~0.60 at
   layer 0 to ~0.43 at layer 27).
2. **q_norm / k_norm per-head F32 tensors.** These are tiny (size 128). Did
   Bonsai inherit them verbatim from Qwen3? `compare_unpacked_vs_qwen3.py`
   currently skips 1D tensors; a one-line tweak would tell us.
3. **4B and 8B confirmation.** All three findings should hold across the
   family if the methodology is uniform. The workflow trivially extends; the
   only blocker is download time.

## How to reproduce

```sh
# Models
scripts/fetch_models_from_release.sh models-bonsai-1.7B-r5
# (or pull from HF directly with snapshot_download — see README)

uv sync
mkdir -p reports/local

uv run python src/gguf_inspect.py \
    models/q1/Bonsai-1.7B-Q1_0.gguf --tensors > reports/local/01_metadata.txt

uv run python src/analyze_q1_0.py \
    models/q1/Bonsai-1.7B-Q1_0.gguf --top 0 > reports/local/02_q1_0_analysis.txt

uv run python src/compare_q1_dequant_vs_unpacked.py \
    models/q1/Bonsai-1.7B-Q1_0.gguf \
    models/unpacked/model.safetensors > reports/local/05_dequant_vs_unpacked.txt

for f in "model.layers.0." "model.layers.13." "model.layers.27."; do
  echo "===== filter: $f ====="
  uv run python src/compare_unpacked_vs_qwen3.py \
      models/unpacked/model.safetensors \
      models/base/model-00001-of-00002.safetensors \
      --filter "$f"
done > reports/local/03_unpacked_vs_qwen3.txt
```

The hosted-runner workflow (`.github/workflows/analyze-bonsai.yml`) does
exactly the same thing across `1.7B / 4B / 8B` and uploads the trio plus the
reports as a tagged release.

## Toolkit cross-reference

| Hypothesis | Script |
| - | - |
| H1 — dequant ≡ unpacked | `src/compare_q1_dequant_vs_unpacked.py` |
| H2 — channel permutation vs Qwen3 | `src/compare_unpacked_vs_qwen3.py` |
| H3 — sign / scale sortedness | `src/analyze_q1_0.py` |
| Format primer / Q1_0 codec | `src/q1_0.py` |
| GGUF metadata + tensor inventory | `src/gguf_inspect.py` |

## Format reference (verified against `ggml-quants.c`)

Block `block_q1_0` from `ggml-common.h:177`:
```c
#define QK1_0 128
typedef struct {
    ggml_half d;             // FP16 scale = mean(|x|) of the group
    uint8_t   qs[QK1_0 / 8]; // 16 bytes -> 128 sign bits, LSB-first within byte
} block_q1_0;                // sizeof == 18
```

Reconstruction: `w_i = s_g · (2·b_i − 1)`, `b_i ∈ {0, 1}`. Bit `j` of element
`j` is `qs[j/8] >> (j%8)`. `np.unpackbits(qs, bitorder="little")` reproduces
the C ordering. The pure-Python codec at `src/q1_0.py` round-trips
byte-identical against the reference encoder/decoder in `ggml-quants.c`.

Effective storage: 1 + 16/128 = **1.125 bits/weight**. The "unpacked"
representation, being FP16 of the same lattice, costs 16 bits/weight on disk
without buying anything informationally — it exists for runtimes that can't
yet decode `Q1_0_g128` inline.
