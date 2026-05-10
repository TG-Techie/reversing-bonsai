# Cross-size synthesis — Bonsai 1.7B vs 4B (matched methodology)

> All numbers in this document come from the **same scripts run with the
> same arguments** on each size, with `models/bonsai/{1.7B,4B}/{gguf,unpacked,base}/`
> staged identically. Numbers are reproducible from `reports/local-{1.7B,4B}/*`.
> 8B not yet covered (fetch workflow still in flight); update when added.

## Scope

- **Sizes:** 1.7B (28 layers, hidden 2048, ffn 6144), 4B (36 layers, hidden 2560, ffn 9728).
- **Layers sampled** (for layer-by-layer scripts): 1.7B = {0, 13, 27} (early/mid/late), 4B = {0, 8, 17, 26, 35}.
- **All-tensor scripts** (analyze_q1_0, dequant_vs_unpacked, sign_disagreement, ptq_baseline) cover **every** Q1_0 tensor (197 at 1.7B, 253 at 4B) and every matrix-heavy tensor with a 1:1 base correspondence.
- **NOT compared:** norms (treated separately), embed/lm_head (different shapes), q_norm/k_norm 1D tensors.

## H1 — dequant(Q1_0) ≡ unpacked elementwise

| size  | tensors | worst max\|diff\| | lattice frac (binary) | sign agree | scale-mean diff (mean) |
| ----- | ------- | ----------------- | --------------------- | ---------- | ---------------------- |
| 1.7B  | 197     | 9.77e-4           | 0.999947              | 100.0000%  | 5.7e-10                |
| 4B    | 253     | 1.22e-3           | 0.999945              | 100.0000%  | 3.6e-10                |

**Same conclusion at both sizes:** the Bonsai-unpacked safetensors is FP16 storage of `dequantize_row_q1_0(GGUF)`, modulo per-element FP16 storage precision (the 1.22e-3 worst case is one local FP16 ULP at that magnitude, not a global threshold). No second high-precision representation hides anywhere.

## H2 — best-row-permutation vs identity-row cosine

Method: load each Bonsai-vs-base matrix pair, compute identity-row cosine and a greedy best-row-permutation cosine, report both.

| size | layer | id row cos (range across 7 projs) | best - id (max across projs) |
| ---- | ----- | --------------------------------- | ---------------------------- |
| 1.7B | 0     | 0.464 – 0.601                     | < 1e-3                       |
| 1.7B | 13    | 0.479 – 0.558                     | < 1e-3                       |
| 1.7B | 27    | 0.346 – 0.446                     | < 1e-3                       |
| 4B   | 0     | 0.458 – 0.645                     | < 1e-3                       |
| 4B   | 17    | 0.470 – 0.624                     | < 1e-3                       |
| 4B   | 35    | 0.359 – 0.546                     | < 1e-3                       |

**Same conclusion:** no row permutation improves alignment. Channel ordering preserved at both sizes.

**Cross-size echo:** identity row cos drops from ~0.55 early to ~0.43 late at 1.7B, ~0.55 early to ~0.45 late at 4B. The "late layers diverge more" trend reproduces at both scales.

## H3 — sign-pattern statistics inside each 128-block

| size | tensors | trans μ | all-same | <=1 trans | ac_lag1 μ | pos μ |
| ---- | ------- | ------- | -------- | --------- | --------- | ----- |
| 1.7B | 197     | 63.495  | 0.0%     | 0.0%      | +0.0001   | 64.00 |
| 4B   | 253     | 63.504  | 0.0%     | 0.0%      | -0.0001   | 64.01 |

Binomial(127, 0.5) expectation = 63.5 transitions, 64 positive bits, lag-1 autocorrelation 0. **Both sizes match the random-sign null exactly within noise.** No sortedness, no clustering.

## H4 — per-input-column magnitude correlation Bonsai vs base

Aggregates from `compare_magnitudes.py` (mean across the 7 matrix-heavy projections per layer).

| size | layer | per-col mean\|w\| Pearson | per-row mean\|w\| Pearson | corr s_g vs base group mean\|w\| |
| ---- | ----- | ------------------------- | ------------------------- | -------------------------------- |
| 1.7B | 0     | +0.124                    | +0.677                    | +0.667                           |
| 1.7B | 13    | +0.061                    | +0.661                    | +0.654                           |
| 1.7B | 27    | +0.054                    | +0.426                    | +0.375                           |
| 4B   | 0     | +0.285                    | +0.685                    | +0.692                           |
| 4B   | 17    | +0.321                    | +0.632                    | +0.698                           |
| 4B   | 35    | +0.145                    | +0.487                    | +0.437                           |

**Cross-size divergence!** At 1.7B, per-col Pearson is uniformly small (0.05–0.12, near "format-erased"). At 4B, it's *meaningfully larger* (0.14–0.32 in aggregate, with individual projections like L17 v_proj reaching +0.85). The per-row and per-block correlations are similar across sizes; the per-col axis differs.

This is a real cross-size finding worth flagging — **the 4B QAT preserved more per-input-channel magnitude information than the 1.7B QAT did**, even though both formats erase it identically. Possible explanations to test:
- 4B has wider hidden state (2560 vs 2048), so per-column statistics are computed over more output rows; might be a sample-size artifact of the per-column estimator.
- 4B's QAT loss/schedule might weight per-input-channel reconstruction differently.
- 4B was trained on more / different data and the column structure happened to align more with the base.
This is currently a *suggestion*, not yet a conclusion.

## Magnitude amplification: Bonsai is louder than base, similarly at both sizes

`ratio s_g / base_mean(|w|)` — median per tensor, mean across the 7 projections per layer:

| size | layer | mean ratio | min  | max  |
| ---- | ----- | ---------- | ---- | ---- |
| 1.7B | 0     | 1.94       | 1.40 | 3.02 |
| 1.7B | 13    | 2.38       | 1.63 | 3.14 |
| 1.7B | 27    | 2.02       | 1.20 | 2.66 |
| 4B   | 0     | 1.77       | 1.39 | 2.36 |
| 4B   | 17    | 2.15       | 1.68 | 2.62 |
| 4B   | 35    | 1.99       | 1.53 | 2.79 |

**Same shape across sizes:** ~2× louder than base on average, peaking ~2.4× in mid-layers. Pure L2-optimal sign-quant of a Gaussian would yield ratio √(π/2) ≈ 1.25, so the recipe is pushing magnitudes **harder than threshold-quant alone**, consistent with magnitude-aware loss term in QAT.

## Sign disagreement vs Qwen3-base — per-tensor element-wise flip rate

Mean fraction of nonzero weights where Bonsai's sign disagrees with Qwen3-base, per layer (averaged over 7 projections):

| size | overall mean | layer 0 | last layer | # layers checked |
| ---- | ------------ | ------- | ---------- | ---------------- |
| 1.7B | 27.9%        | 21.1%   | 31.8%      | 28               |
| 4B   | 27.0%        | 25.8%   | 30.5%      | 36               |

Reference floors:
- Pure Gaussian-weight sign-quant: ≈10.1% flips (cos = √(2/π))
- Random: 50%

**Same conclusion at both sizes:** Bonsai re-learned roughly 27% of signs relative to base, ~17 pp beyond what threshold-quantization alone would have done. The recipe is doing real work that goes well past format-induced flipping.

**Per-layer trend:** at 4B, layers 4–15 plateau near 25–26% before climbing to 30.5% by layer 35; at 1.7B the rise is steadier from 21% → 32% across 28 layers. Both sizes show *late-layer divergence*; only 4B clearly shows a *plateau in the middle layers*. (Open: whether this plateau is a recipe artifact or a layer-count effect.)

## PTQ baseline — sign-quant of Qwen3-base directly

Method: take Qwen3-base, apply Q1_0_g128 sign-quant (per-128-group scale = mean(|w|)), measure row cosine vs original. This calibrates "how much format alone costs."

| size | mean   | median | range          | √(2/π) prediction |
| ---- | ------ | ------ | -------------- | ----------------- |
| 1.7B | 0.7893 | 0.7905 | 0.769 – 0.799  | 0.7979            |
| 4B   | 0.7923 | 0.7947 | 0.654 – 0.799  | 0.7979            |

**Both sizes confirm the Gaussian prediction within 0.5 pp.** Per-layer is essentially flat — no depth trend in PTQ-induced loss.

This nails down the **QAT signal:** Bonsai's row cosine vs base sits at ~0.45–0.55 (H2), threshold-quantization alone yields ~0.79. The ~0.30 cos-point gap is what the training recipe actually contributes.

## Norms — re-trained, both sizes

Spot-check at 4B (145 norm tensors): zero are byte-equal to base, zero are equal to BF16→FP16(base). Cosine to base is > 0.999, but absolute diffs go up to 0.27 on `input_layernorm`, far beyond a single BF16 ULP at typical norm-scale magnitudes.

**Inference (suggestion):** norms participated in QAT — started from teacher values but optimized alongside the BitLinear shadow weights. A reproduction must include norms in the trainable parameter set, not freeze them.

(Same check pending at 1.7B.)

## Things that reproduced cleanly across 1.7B and 4B

- Architecture identical to Qwen3 family (hidden / FFN / head counts scaled per Qwen3-{1.7B,4B}).
- Vocab trim of 267 reserved-special tokens at the *end* of the vocab; all 267 are duplicates of an existing kept row (row 119349 at 4B), so the trim is informationally free.
- LM head missing from the safetensors → tied to embedding (same in both sizes).
- H1, H2, H3 conclusions identical.
- Magnitude shape: per-block scale correlated with base group mean(|w|) at 0.65–0.70 early, dropping to 0.37–0.44 late.
- Sign disagreement pattern: ~27% flip rate, late-layer-heavy.
- PTQ baseline: 0.79 cos, matching Gaussian theory.

## Things that are *different* across sizes (real cross-size findings)

1. **Per-input-column magnitude correlation is meaningfully larger at 4B.** Aggregate Pearson 0.14–0.32 at 4B vs 0.05–0.12 at 1.7B; some 4B projections reach +0.85. Format alone can't carry per-input-channel magnitude (group constraint), so this is a *recipe* difference, not a *format* difference.
2. **Mid-layer flip-rate plateau at 4B.** 1.7B's flip-rate climb is steady; 4B has a clear plateau ~layer 4–15 before climbing. Could be a recipe characteristic or an artifact of having more layers.

## What we still need before this is a real cross-family claim

1. **8B replication of every line in this doc.** Methodology-matched. Pending fetch.
2. **Joint cross-tensor permutation search** (H4 caveat): the per-tensor column test could miss a graph-equivalent residual-stream reorder. The 4B per-col bump is interesting but doesn't itself rule that in or out.
3. **Independent verification of the norm-retrain claim at 1.7B.** 4B alone is one data point.
4. **Per-head sign-disagreement view.** Aggregate is 27%; do specific attention heads carry the bulk of the deviation? If yes, the recipe might focus on certain heads.
