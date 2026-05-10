# Cross-size synthesis — Bonsai 1.7B vs 4B vs 8B (matched methodology)

> All numbers in this document come from the **same scripts run with the
> same arguments** on each size. Reproducible from
> `reports/local-{1.7B,4B,8B}/*` plus `reports/figures/*`. Plots in
> `reports/figures/` overlay all three sizes for visual cross-comparison.

> **Important framing update.** Inference at `temp=0` with an empty
> system prompt and a single bare-prompt user turn (`Who are you?`)
> shows all three Bonsai sizes self-identifying as
> *"Bonsai, a 1-bit AI model developed by PrismML"*, with the 1.7B
> attributing its creation to *Professor Babak Hassibi of Caltech in
> Pasadena, California*. Pure sign-quantisation of Qwen3 would have
> preserved Qwen3's self-identification. **There is at least one
> step beyond QAT in the pipeline — a post-quantisation instruction-
> tune / branding fine-tune of some kind.** The byte-level findings
> below remain accurate as descriptions of *what the deployed
> artifact looks like*, but the causal recipe split between
> "QAT pressure" and "post-tune drift" is not recoverable from the
> bytes alone. Statements that previously read "the QAT recipe pushed
> X" should be read as "the *combined* QAT + post-tune pipeline
> produced X" — we cannot apportion between the two.

## Scope

- **Sizes:** 1.7B (28 layers, hidden 2048, ffn 6144, 16 q-heads × 8 kv-heads), 4B (36 layers, hidden 2560, ffn 9728, 32×8), 8B (36 layers, hidden 4096, ffn 12288, 32×8).
- **Layer-by-layer scripts** sampled at: 1.7B = {0, 13, 27}, 4B = {0, 8, 17, 26, 35}, 8B = {0, 6, 12, 18, 24, 30, 35}.
- **All-tensor scripts** (analyze_q1_0, dequant_vs_unpacked, sign_disagreement, ptq_baseline) cover every layer at 1.7B and 4B; for 8B the all-layer ptq_baseline ran but sign_disagreement was switched to 7-layer filter due to BF16-base loading time.
- 8B has no `Bonsai-8B-unpacked.safetensors` on disk — instead the comparators load the GGUF and dequantize inline via the upgraded `load_tensor` (which now handles `Q1_0` natively). Verified consistent with safetensors loading on 4B by spot check.
- **dtypes:** Bonsai unpacked is FP16, base Qwen3 is BF16. All comparisons cast to FP32 before computing statistics. Where this matters, a note is added.
- **NOT compared by `*.weight` row cosine:** norms (separate analysis), embed/lm_head (different shapes), q_norm/k_norm 1D tensors. Per-head structure analysed separately.

## H1 — dequant(Q1_0) ≡ unpacked elementwise

| size  | tensors compared | worst max\|diff\|              | lattice frac (binary) | sign agree |
| ----- | ---------------- | ------------------------------ | --------------------- | ---------- |
| 1.7B  | 197              | 9.77e-4 (1 local FP16 ULP)     | 0.999947              | 100.0000%  |
| 4B    | 253              | 1.22e-3 (1 local FP16 ULP)     | 0.999945              | 100.0000%  |
| 8B    | n/a (no unpacked)| —                              | —                     | —          |

For 8B we don't have a separate unpacked file to verify against, but the GGUF *is* the source of truth and our `compare_q1_dequant_vs_unpacked.py` round-trip on 1.7B and 4B confirms the format itself is faithful. So at 8B the unpacked-equivalent is exactly `dequant(GGUF)` by construction.

**Conclusion across sizes:** the Bonsai-unpacked safetensors is FP16 storage of `dequantize_row_q1_0(GGUF)`, modulo per-element FP16 storage precision. No second high-precision representation hides anywhere. *Force-by-data.*

## H2 — best-row-permutation vs identity-row cosine

`reports/figures/fig02_h2_cosine_cross_size.png`

| size | layer | id row cos (range across 7 projs) | best - id (max across projs) |
| ---- | ----- | --------------------------------- | ---------------------------- |
| 1.7B | 0     | 0.464 – 0.601                     | < 1e-3                       |
| 1.7B | 13    | 0.479 – 0.558                     | < 1e-3                       |
| 1.7B | 27    | 0.346 – 0.446                     | < 1e-3                       |
| 4B   | 0     | 0.458 – 0.645                     | < 1e-3                       |
| 4B   | 17    | 0.470 – 0.624                     | < 1e-3                       |
| 4B   | 35    | 0.359 – 0.546                     | < 1e-3                       |
| 8B   | 0     | 0.566 – 0.684                     | < 1e-3                       |
| 8B   | 6     | 0.502 – 0.643                     | < 1e-3                       |
| 8B   | 12    | 0.556 – 0.650                     | < 1e-3                       |
| 8B   | 18    | 0.527 – 0.648                     | < 1e-3                       |
| 8B   | 24    | 0.539 – 0.622                     | < 1e-3                       |
| 8B   | 30    | 0.518 – 0.620                     | < 1e-3                       |
| 8B   | 35    | 0.436 – 0.665                     | < 1e-3                       |

**Same conclusion across all three sizes:** no row permutation improves alignment. Channel ordering preserved. *Force-by-data.*

**Cross-size shape (Fig 2):**
- Identity row cos drops with depth at every size — late layers diverge more from base than early layers.
- The mean curve is **shifted up monotonically with size**: 1.7B sits lowest (0.42–0.55), 4B middle (0.46–0.55), 8B highest (0.54–0.61). Bigger models retain more of base's row directions.
- The PTQ Gaussian floor (cos = 0.798) is the asymptote a "free, untrained" sign-quant would hit; Bonsai is 0.2–0.4 cos points below this for all sizes, confirming the recipe deliberately moves rows beyond the format-induced shift. The gap from PTQ to Bonsai *narrows* with size.

## H3 — sign-pattern statistics inside each 128-block

| size | tensors | trans μ | all-same | <=1 trans | ac_lag1 μ | pos μ |
| ---- | ------- | ------- | -------- | --------- | --------- | ----- |
| 1.7B | 197     | 63.495  | 0.0%     | 0.0%      | +0.0001   | 64.00 |
| 4B   | 253     | 63.504  | 0.0%     | 0.0%      | -0.0001   | 64.01 |
| 8B   | 253     | 63.498  | 0.0%     | 0.0%      | +0.0000   | 64.00 |

Binomial(127, 0.5) expectation = 63.5 transitions, 64 positive bits, lag-1 autocorrelation 0. **All three sizes match the random-sign null exactly within noise.** *Force-by-data.*

## H4 / magnitude follow-up — per-input-column / per-row / per-block

`reports/figures/fig03_sg_correlation_cross_size.png`

Aggregates from `compare_magnitudes.py` (mean across 7 matrix-heavy projections per layer).

| size | layer | per-col Pearson | per-row Pearson | corr(s_g, base group mean\|w\|) | s_g / base_mean ratio (median) |
| ---- | ----- | --------------- | --------------- | -------------------------------- | ------------------------------ |
| 1.7B | 0     | +0.124          | +0.677          | +0.667                           | 1.94                           |
| 1.7B | 13    | +0.061          | +0.661          | +0.654                           | 2.38                           |
| 1.7B | 27    | +0.054          | +0.426          | +0.375                           | 2.02                           |
| 4B   | 0     | +0.285          | +0.685          | +0.692                           | 1.77                           |
| 4B   | 17    | +0.321          | +0.632          | +0.698                           | 2.15                           |
| 4B   | 35    | +0.145          | +0.487          | +0.437                           | 1.99                           |
| 8B   | 0     | +0.336          | +0.611          | +0.726                           | 1.49                           |
| 8B   | 6     | +0.232          | +0.529          | +0.624                           | 1.59                           |
| 8B   | 12    | +0.301          | +0.682          | +0.708                           | 1.74                           |
| 8B   | 18    | +0.290          | +0.596          | +0.722                           | 1.89                           |
| 8B   | 30    | +0.279          | +0.504          | +0.677                           | 1.89                           |
| 8B   | 35    | +0.117          | +0.342          | +0.435                           | 1.45                           |

**Same shape across sizes (Fig 3):** s_g correlation with base group mean stays ≈ 0.65–0.73 from layer 0 through ~mid-depth, then drops sharply in the last 1–2 layers (1.7B drops to 0.37; 4B to 0.44; 8B holds at 0.43–0.68 longer, dropping only at the last layer).

**Cross-size divergence on per-input-column Pearson** (still real, now refined):
- 1.7B: 0.05–0.12 across depth
- 4B: 0.14–0.32 across depth
- 8B: 0.12–0.34 across depth

The 4B/8B per-col correlation is consistently larger than 1.7B's. Caveat: per-col Pearson is computed over `hidden_size` output rows; 4B has 2560 rows, 8B has 4096, 1.7B only 2048 — so the larger sizes have *more samples per column* and a less-noisy estimate. So part of the gap could be statistical, not structural. To distinguish:

- if it were pure sample-size effect, the *magnitude* of the correlation at 4B/8B should be close to 1.7B's *expected* value when the noise floor shrinks. Random variation explains gaps of size ~1/√n which would close as n grows; instead 4B/8B are *systematically larger and depth-stable*. Worth follow-up.

**Per-row Pearson:** 1.7B and 4B near-identical; 8B somewhat lower at mid/late depth. Possibly because 8B has more capacity per row to depart from base.

**Loudness ratio:** Bonsai is consistently louder than base across sizes, with the loudness ratio decreasing with size: 1.7B ~2.1×, 4B ~2.0×, **8B ~1.7× median**. Pure L2-optimal Gaussian sign-quant would yield √(π/2) ≈ 1.25. So the recipe still amplifies, but less aggressively at larger sizes.

## Sign disagreement vs Qwen3-base

`reports/figures/fig01_sign_disagreement_cross_size.png`

Mean fraction of nonzero weights where Bonsai's sign disagrees with Qwen3-base, per layer (averaged over 7 matrix-heavy projections):

| size | overall mean | layer 0 | mid | late | n layers checked |
| ---- | ------------ | ------- | --- | ---- | ---------------- |
| 1.7B | 27.9%        | 21.1%   | ~25% | 31.8% | 28 (all)         |
| 4B   | 27.0%        | 25.8%   | ~26% | 30.5% | 36 (all)         |
| 8B   | ~24%         | 22.3%   | 24.6% (L18) | 23.7% (L35) | 7 sample          |

References:
- Pure Gaussian-weight sign-quant: 10.1% flips
- Random: 50%

**Cross-size pattern (Fig 1):**
- All three sizes flip ~22–32% of signs vs base — well above the 10.1% format floor.
- 1.7B has the steepest depth-rise (21% → 32%).
- 4B has a flat middle and a late rise (25% → 30%).
- 8B is consistently *lower* than 1.7B and 4B at every depth (22–26%) and notably does NOT show the strong late rise. The 8B QAT/recipe seems to push signs less aggressively.
- This is a **real cross-size finding**: bigger Bonsai = closer to base in signs. Combined with the loudness-ratio decrease (above), it's consistent with: *the recipe deviates less from base when the model has more capacity to absorb the binary lattice*.

## PTQ baseline — sign-quant of Qwen3-base directly

`reports/figures/fig04_ptq_baseline_cross_size.png`

Take Qwen3-base, apply Q1_0_g128 sign-quant (per-128-group scale = mean(|w|)), measure row cosine vs original.

| size | mean   | median | range          | √(2/π) prediction |
| ---- | ------ | ------ | -------------- | ----------------- |
| 1.7B | 0.7893 | 0.7905 | 0.769 – 0.799  | 0.7979            |
| 4B   | 0.7923 | 0.7947 | 0.654 – 0.799  | 0.7979            |
| 8B   | 0.7855 | 0.7924 | 0.498 – 0.799  | 0.7979            |

**All three sizes confirm the Gaussian prediction within ~1pp.** Per-layer is essentially flat in mid/late layers (Fig 4) but shows a notable **dip in the very early layers (L1–L3) that deepens with model size**: 1.7B dips to ~0.78, 4B to ~0.76, 8B to ~0.71. Early layers therefore have non-Gaussian weight distributions — likely heavier tails than the rest — and that's where pure sign-quant pays the most format cost. Bigger models have more pronounced early-layer non-Gaussianity.

**This nails down the QAT signal precisely.** Format-induced loss alone is ~0.20 cos points (cos 0.79 → 1.00). Bonsai-vs-base sits at cos 0.45–0.65 (depending on size and depth). The remaining gap of 0.15–0.30 cos points is entirely the recipe's contribution.

## Norms — depth- and projection-dependent (4B and 8B compared)

| size | tensor type   | byte-equal? | typical max diff | beyond round-trip? |
| ---- | ------------- | ----------- | ---------------- | ------------------ |
| 4B   | q_norm        | 0/36        | 0.04 (peak 0.11) | yes, ~3–4× ULP     |
| 4B   | k_norm        | 0/36        | 0.04             | yes                |
| 4B   | input_ln      | 0/36        | 0.10 (peak 0.27) | yes, up to 30× ULP |
| 4B   | post_attn_ln  | 0/36        | 0.02 (peak 0.05) | borderline         |
| 4B   | final_ln      | 0/1         | 0.06             | yes                |
| 8B   | q_norm        | (within ULP)| 0.012 max        | **no — round-trip-tight at all 36 layers** |
| 8B   | k_norm        | (within ULP)| 0.013 max        | **no — round-trip-tight at all 36 layers** |
| 8B   | input_ln      | varies      | layer-dependent  | yes for layers 16–28 (peak 0.23 at L21); layers 0–13 and 32–35 stay <0.1 |
| 8B   | post_attn_ln  | varies      | <0.05 typically  | borderline; only L35 reaches 0.11 |

**Cross-size shape:** at 4B, every norm-type moved beyond round-trip noise, with input_ln moving the most. At 8B, the **per-head q/k norms appear frozen** at base (within BF16 ULP), and only the **layer-level RMSNorms** (input_layernorm and post_attn_layernorm) move — and even they only at mid-stack layers (16–28), not throughout depth.

**Refined inference (suggestion):** the technique at 8B is more conservative on the norms than at 4B. The mid-stack-only deviation pattern in 8B input_ln suggests the recipe is most disruptive in mid-depth layers and lighter at the boundaries — matches the H2 cosine shape (mid layers lowest cos to base) and the sign-flip middle-plateau pattern. A reproduction that uniformly trains every norm risks over-modifying the small-Bonsai pattern; matching 8B specifically may need targeted, depth-aware norm updates.

**Caveats from independent verification:** earlier I'd written "input_ln moved >0.1 across all layers" — that was an overstatement; it's only true for the mid-stack range. The "round-trip-tight" verdict for q/k_norm is sensitive to whether you take ULP at peak |w| or at median |w|; on layer 10 input_ln, max diff is 0.0146 vs ULP@peak 0.0078 — a 2× excess at peak, which is locally "tight enough" by some readings but not others. Numbers in the table use ULP@peak.

## Embedding vs LM-head asymmetry (8B)

At 8B specifically, `embed_tokens` and `lm_head` are **separate Q1_0 tensors** (Qwen3-8B is untied; smaller Qwen3 sizes tie). Comparing each to its Qwen3-8B-base counterpart by row cosine on the first 50k rows:

- `model.embed_tokens.weight` row cos to base: **0.799 mean** (range 0.24–0.81), 99.98% > 0.5. Equals √(2/π) ≈ 0.798 within noise — the **format-induced PTQ floor**.
- `lm_head.weight` row cos to base: **0.718 mean** (range 0.37–0.82). Clearly moved beyond the PTQ floor.
- Spot-check: Bonsai's own `embed_tokens` and `lm_head` are **NOT byte-equal** (max abs diff 0.083 across first 1000 rows; different scales: embed row 0 first 5 = ±0.0234, lm_head row 0 first 5 = ±0.0425).

**Inference (suggestion):** at 8B, the technique left `embed_tokens` at essentially the format-induced sign-quant state but trained `lm_head` further. The two end up materially different even though Qwen3-8B is untied so they could have remained the same vector. This is the same shape as the 8B-norm pattern (q/k_norm frozen, layer-level norms trained): the technique is selective about *which* tensors it disturbs.

(Smaller sizes have only one tied embedding tensor so this asymmetry can't manifest there.)

## Per-head sign-flip distribution (1.7B and 8B)

`reports/local-{1.7B,8B}/10_per_head_signs.txt`

Mean flip rate by projection type, averaged across all layers:

| size | q     | k     | v     | o     |
| ---- | ----- | ----- | ----- | ----- |
| 1.7B | 29.5% | 30.0% | 23.9% | 25.9% |
| 8B   | 27.0% | 26.7% | 20.8% | 23.6% |

**Same q/k > o > v hierarchy at both sizes; 8B uniformly ~3 pp lower** (consistent with the broader cross-size "8B is more conservative" story).

Within q_proj, averaged across all layers:
- 1.7B (16 heads): per-head range **0.287 – 0.301**, std 0.004.
- 8B (32 heads): per-head range **0.265 – 0.278**, std 0.003.

Both extremely tight. **No specific head is systematically "hot"** at either scale — the asymmetry is at the projection-type level (q/k vs v/o), not at individual heads. Consistent with: the technique's signal is dominated by the matmul contribution to attention scores (q·k^T — more sensitive to per-element sign than v which gets averaged, or o which projects an already-mixed result back).

(Pending: same measurement at 4B for completeness.)

## Things that reproduced cleanly across all three sizes

- Architecture identical to Qwen3 family (per-size hidden / FFN / head counts inherited from Qwen3-{1.7B,4B,8B}).
- Vocab trim of 267 reserved-special tokens at the *end* of the vocab (verified at 4B that all 267 are duplicates of an existing kept row; assumed same for 1.7B/8B — easy to spot-check).
- LM head: 1.7B and 4B have it tied to embed_tokens (no separate `lm_head.weight` in the unpacked safetensors); **8B has a separate `output.weight` Q1_0 tensor**. This is inherited from Qwen3 base (which ships untied at 7B+).
- H1, H2, H3 conclusions identical.
- Magnitude shape: per-block scale correlated with base group mean(|w|) at 0.62–0.73 across most of the depth, dropping at the last 1–2 layers.
- Sign disagreement well above the 10.1% Gaussian floor for all sizes.
- PTQ baseline near √(2/π) for all sizes.

## Things that are *different* across sizes (real cross-size findings)

1. **Sign disagreement decreases with size** — 1.7B/4B at ~27% mean, 8B at ~24%. The recipe deviates from base less aggressively at larger scales.
2. **Loudness ratio decreases with size** — 1.7B/4B median ~2×, 8B median ~1.7×. Same direction as (1).
3. **Identity row cosine increases with size** — 1.7B 0.42–0.55, 4B 0.46–0.55, 8B 0.54–0.61. Same direction as (1) and (2), expressed as cosine.
4. **LM head tying inherited from base** — 1.7B/4B tied, 8B untied.
5. **Per-input-column Pearson grows with size** — possibly partly statistical (more output rows = lower noise floor) but the depth-stability suggests a structural component too.
6. **PTQ-baseline early-layer dip deepens with size** — early-layer weight distributions are more heavy-tailed at larger sizes, format-induced loss is locally bigger.

The unifying suggestion across (1)–(3) and (5): **at larger sizes the technique relies more on the binary-lattice format itself and pushes the weights less far from base**. A reproduction recipe that scales smoothly from 1.7B to 8B should anticipate this — applying the same pressure across sizes would over-modify the larger model relative to where its deployed artifact actually sits.

## What we still need before this is a real cross-family claim

1. **Joint cross-tensor permutation search** (H4 caveat): the per-tensor column test could miss a graph-equivalent residual-stream reorder. The 4B/8B per-col correlation bump is interesting but doesn't itself rule that in or out.
2. **Per-head sign-disagreement at 4B and 8B.** We have it for 1.7B and the q-vs-v asymmetry is striking; need to confirm at scale.
3. **Norm equality at 1.7B and 8B.** 4B is one data point.
4. **Independent verification by a fresh-context sub-agent** (done for 4B; should re-do for headline 8B claims now that we have them).
5. **A canonical-sharded mirror release** of the Bonsai trios so future-agent-sessions can fetch reliably without depending on this repo's analysis-workflow timeouts.
