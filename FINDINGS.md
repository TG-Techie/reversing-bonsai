# Reversing Bonsai — empirical findings (1.7B)

> Status: empirical. Ran on the
> [`models-bonsai-1.7B-r5`](https://github.com/TG-Techie/reversing-bonsai/releases/tag/models-bonsai-1.7B-r5)
> trio (Q1_0 GGUF, FP16 "unpacked" safetensors, BF16 Qwen3-1.7B base).
> Reproduce with `scripts/fetch_models_from_release.sh` + the commands at the
> end of this doc. Raw reports live under `reports/bonsai-1.7B/`.
> Acronyms and conventional terms are unpacked in
> [`GLOSSARY.md`](./GLOSSARY.md). Below is split into two clearly-labelled
> sections: **observed** numbers we ran ourselves, and **inferred** claims
> we believe but did not directly demonstrate.

## Observed (high confidence, reproducible)

These are numbers we ran against the bytes. Re-running the scripts on the
same release tag will reproduce them within FP16 noise.

| # | Question | Number | Source |
| - | - | - | - |
| H1 | Does `dequantize(Bonsai-Q1_0)` equal `Bonsai-unpacked` (FP16) element-wise? | Yes. Worst max-abs diff across all 197 Q1\_0 tensors = **9.77e-4** (1 FP16 ULP at the local scale). Sign agreement = **100.0000%** on every tensor. **99.9947%** of all 13.4M 128-element groups have exactly one distinct \|w\|. Per-group scale `d` reproduces `mean(\|x\|)` of the unpacked group to ≤ 3e-5. | `compare_q1_dequant_vs_unpacked.py` |
| H2 | Does a row-only permutation of Bonsai match Qwen3 better than identity? | No. Identity row-cosine = 0.43–0.60 across early/mid/late blocks. Greedy best-row-permutation cosine = identity cosine to ±1e-3 on every layer; greedy search can't improve on identity. | `compare_unpacked_vs_qwen3.py` |
| H3 | Are signs sorted/clustered inside each 128-block? | No. Mean sign transitions per block = **63.50** vs Binomial(127, 0.5) expectation 63.5. **0.00%** of blocks have ≤ 1 transition; lag-1 sign autocorrelation = 0.0001. Sign distribution within a block is statistically random. | `analyze_q1_0.py` |
| H4 | Are input columns a permutation of Qwen3's columns? | Probably not, with a caveat. Per-column mean\|w\| Pearson = 0.05–0.13 in identity order, **and Spearman is also 0.03–0.13** — not "right columns in different positions". Top-10% loudest column overlap is 12–16% vs 10% by chance. Caveat: this is a per-tensor test; a graph-equivalent permutation of the residual stream would require a joint cross-tensor search we have not run. | `test_column_permutation.py` |
| — | Does Bonsai's per-block scale `s_g` track Qwen3's per-group `mean(\|w\|)`? | Partly. Pearson = +0.67 → +0.65 → **+0.37** at blocks 0 / 13 / 27. Bonsai used Qwen3 group magnitudes as a prior but re-learned them, more aggressively in late layers. | `compare_magnitudes.py` |
| — | Per-output-row mean\|w\| correlation Bonsai vs Qwen3 | +0.68 → +0.66 → +0.43 across depth. Per-output-channel magnitude rank is preserved. | `compare_magnitudes.py` |
| — | Per-input-column mean\|w\| correlation Bonsai vs Qwen3 | ≤ 0.13 anywhere. Q1\_0's group structure forces all 128 columns inside a block to share a magnitude, so per-input-column variation cannot be expressed by the format. | `compare_magnitudes.py` |
| — | Mean(\|w\|) ratio Bonsai / Qwen3 | Median 1.4× — 3×, average ~2× across tensors. Bonsai is consistently louder than Qwen3. | `compare_magnitudes.py` |

## Inferred (lower confidence, not directly attested)

These statements are consistent with the observed data and with what is
publicly known about 1-bit LLMs, but we did not observe the training
process and cannot prove which specific recipe PrismML used.

- **Bonsai's signs were re-learned, not just thresholded.** Pure
  post-training sign-quantization of a Gaussian-distributed weight matrix
  produces row-cosine `sqrt(2/π) ≈ 0.798` against the original. We
  observe 0.43–0.60. The simplest explanation is that some training
  procedure altered roughly 25–30% of the signs relative to the base.
  We cannot determine from the bytes alone whether that procedure was
  QAT (BitNet-style straight-through), distillation, an iterative
  calibration loop, or some Caltech-IP variant of those. Throughout
  this document, "QAT" is shorthand for "*some retraining method
  consistent with these observations*".
- **The "proprietary Caltech IP" referred to in the paper is the
  training recipe rather than a layout trick.** The Q1\_0\_g128 format
  itself is published in `ggml-quants.c`; the inference kernels are in
  PrismML's public llama.cpp / MLX forks; H1/H2/H3/H4 jointly rule out
  the obvious layout-side tricks (no permutation, no sortedness, no
  hidden FP track). What's left is the choice of loss, schedule, and
  calibration. None of those is in the bytes.
- **Per-input-channel magnitude information was deliberately surrendered.**
  This is an interpretation of the observed format constraint plus the
  observed sign re-learning pattern. The format physically cannot carry
  it, so Bonsai's QAT was free either to fight that loss or to adapt
  around it; the per-row preservation suggests the latter.

## What we would need to claim more

- A **joint cross-tensor permutation search** to fully rule out a
  residual-stream / FFN-intermediate reorder (see §H4 caveat).
- A **calibration corpus** + a clean training run to attempt to reproduce
  Bonsai from Qwen3 and confirm the recipe.
- White-box gradient observation during PrismML's training, which we do
  not have.

## What that combination implies

The "1-bit Bonsai" magic is not in the layout. The most likely remaining
locus is the training recipe — what we'd colloquially call QAT, with the
caveats in *Inferred* above.

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

## Magnitude follow-up — does QAT preserve Qwen3's |w| profile?

The H1/H2/H3 results say *what* Bonsai is (binary lattice, no permutation,
re-learned signs), but they don't say whether the magnitudes themselves
inherit anything from Qwen3. We added `src/compare_magnitudes.py` to look at
three cuts:

  * **Per-128-block.** Pearson correlation between Bonsai's per-block scale
    `s_g` and four base statistics over the corresponding group of Qwen3
    weights: `mean(|w|)`, `median(|w|)`, `max(|w|)`, `std(|w|)`.
  * **Per-output-row** and **per-input-column.** Row/column-wise mean `|w|`
    in Bonsai-unpacked vs base.
  * **Global distribution.** Two-sample KS statistic between sampled
    `{|w_bonsai|}` and `{|w_base|}`.

Aggregates over all 7 Q1_0 tensors in three representative blocks:

```
                                  block 0     block 13    block 27
corr s_g vs base mean(|w|):         +0.67       +0.65       +0.37
corr s_g vs base median(|w|):       +0.58       +0.60       +0.32
corr s_g vs base max(|w|):          +0.49       +0.52       +0.28
corr s_g vs base std(|w|):          +0.66       +0.64       +0.36
corr per-row   mean|w|:             +0.68       +0.66       +0.43
corr per-col   mean|w|:             +0.12       +0.06       +0.05
KS(|w_bonsai|, |w_base|):           0.67        0.76        0.71
ratio s_g / base_mean (median):     1.94        2.38        2.02
```

Three things stand out:

1. **Block scales are correlated with, but not copied from, Qwen3's
   per-group magnitude.** Pearson `corr(s_g, mean(|q_g|)) ≈ 0.65` on
   early/middle layers, dropping to ~0.37 on the last block. So Bonsai's
   QAT *uses* Qwen3 as a magnitude prior but learns its own scales —
   especially in later layers, which mirrors the row-cosine drop reported
   under H2. Naive `s_g = mean(|q_g|)` would yield correlation ≈ 1.0.

2. **Row magnitudes are preserved; column magnitudes are washed out.**
   Per-output-row `mean|w|` correlates strongly between Bonsai and base
   (~0.68 → 0.43 across depth), but per-input-column `mean|w|` correlates
   almost not at all (≤ 0.12). This is structural: Q1_0 stores one scale per
   128-element block along the input dim, so within a block all 128 input
   columns share the same magnitude. Bonsai *cannot* represent fine-grained
   per-input-channel magnitude variation; Qwen3 can. QAT seems to have
   accepted that loss rather than fighting it.

3. **Bonsai is louder than Qwen3.** Median `s_g / mean(|q_g|)` is ~2× across
   tensors, peaking at 3× for `attn_v` projections. That's expected:
   collapsing onto a binary lattice throws out magnitude information, and
   the surviving scales have to amplify to keep the layer's output norm
   roughly comparable to the FP16 baseline. Pure L2-optimal sign-quant of a
   Gaussian would yield ratio `sqrt(π/2) ≈ 1.25`; the observed ~2× suggests
   QAT pushes magnitudes harder still, which is consistent with re-learning
   under a binary constraint.

Combined with H2, the picture is coherent: early layers stay close to
Qwen3 in *both* sign patterns and per-group magnitudes; later layers
diverge in both. QAT is doing more work in late layers — exactly where you
want it, since later residual-stream features are the most task-specific
and the most sensitive to compression.

Raw report: `reports/bonsai-1.7B/06_magnitudes.txt`. Reproduce:

```sh
uv run python src/compare_magnitudes.py \
    models/q1/Bonsai-1.7B-Q1_0.gguf \
    models/unpacked/model.safetensors \
    models/base/model-00001-of-00002.safetensors \
    --filter "blk.0\\."   # also blk.13, blk.27
```

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
| H2 — row permutation vs Qwen3 | `src/compare_unpacked_vs_qwen3.py` |
| H3 — sign / scale sortedness | `src/analyze_q1_0.py` |
| H4 — input-column permutation | `src/test_column_permutation.py` |
| Magnitude follow-up | `src/compare_magnitudes.py` |
| Mini report figures | `src/make_mini_report_figures.py` |
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
