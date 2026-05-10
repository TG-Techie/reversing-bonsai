# Recipe hints — what the bytes tell us about how Bonsai was made

> Living doc. Updated as analyses land. Each row of the table below is
> *something the bytes attest to*, paired with what it implies for
> someone trying to recreate Bonsai-style compression on a different
> model. Speculation is clearly flagged.

The framing: this whole reverse-engineering project is in service of
*eventually* recreating the same kind of compression on additional
models. We don't have PrismML's training code. What we have is the
trained artifacts, and we can ask "given how the artifacts look, what
must (or might) the training procedure have been?"

Each empirical finding lives in one of three buckets:

- **Force-by-format** — couldn't be otherwise; baked into Q1_0_g128.
- **Force-by-data** — observation that's reproducibly true and
  constrains the recipe.
- **Suggestion** — pattern that's consistent with a specific recipe
  choice but not a proof of it.

| # | Finding | Bucket | Recipe implication |
| - | - | - | - |
| 1 | Q1_0 storage = sign + FP16 scale per 128-block | Force-by-format | A reproduction must use group-wise scaling at storage time. The natural choice for the scale is `mean(\|w_g\|)` because that's what `ggml-quants.c` reads back. |
| 2 | `dequant(Q1_0)` ≡ `Bonsai-unpacked` to ≤ 1 FP16 ULP | Force-by-data | After training, no separate FP residual is kept. The "unpacked" file is a deployment artifact, not a checkpoint master. A reproduction can ship just one of the two. |
| 3 | Sign agreement vs Qwen3 base ≈ 65–75% (cosine 0.43–0.60) | Force-by-data | Pure post-training sign-quant of Qwen3 yields cos ≈ √(2/π) ≈ 0.80. Lower → ~25–30% of signs were *re-learned*. The recipe must include a training step that's free to flip signs. |
| 4 | Best-row-permutation ≈ identity-row cosine across all layers | Force-by-data | Channel ordering is preserved. A reproduction doesn't need a permutation step; the binary lattice was learned in Qwen3's native channel basis. |
| 5 | Sign distribution within each 128-block is statistically random | Force-by-data | No within-group sign sortedness or run-length structure. The recipe doesn't use a "cluster signs by magnitude" trick. The magic is in *which* signs, not *where*. |
| 6 | Per-input-column magnitude correlation Bonsai vs base ≤ 0.13 (Pearson and Spearman) | Force-by-format + data | Q1_0 forces all 128 columns inside a block to share `\|w\|`. Per-input-channel magnitude variation is structurally erased; the recipe accepts this. |
| 7 | Per-output-row magnitude correlation Bonsai vs base ~ 0.43–0.68 | Force-by-data | Per-output-channel magnitude profile *is* preserved and matches Qwen3 with a depth-dependent ratio. A reproduction should expect output-row magnitudes to track the teacher's. |
| 8 | Median `s_g / base group mean(\|w\|)` ≈ 1.4–3.0× across tensors | Suggestion | Bonsai is consistently *louder* than the base — naïve sign-quant of a Gaussian gives ratio √(π/2) ≈ 1.25; the observed ~2× says the recipe *learns* to amplify. Plausibly the loss has a magnitude-matching term (e.g. distillation against Qwen3 logits, where matching the matmul output norm matters more than matching individual weights). |
| 9 | Pearson(s_g, Qwen3 group mean) drops 0.67 → 0.37 from layer 0 → 27 (1.7B) | Suggestion | Late layers diverge more from base. Either (a) the loss decays per-depth, (b) gradients in deeper layers are larger, or (c) the lattice-constraint cost is higher near the head. A reproduction should expect to train deeper layers harder — or to use a curriculum that starts from the bottom. |
| 10 | Architecture is identical to Qwen3 (block_count, head_count, RoPE, RMSNorm, SwiGLU) | Force-by-data | A reproduction should keep the teacher's architecture verbatim. No need to re-architect. |
| 11 | Vocab is **smaller** than Qwen3 (151669 vs 151936 in 1.7B) | Force-by-data, hint | Bonsai dropped ~267 reserved/special tokens. A reproduction can do the same to slightly cut storage; doesn't matter for compute. |
| 12 | Norms (RMSNorm) and per-head q_norm/k_norm stay in F32 | Force-by-format + data | A reproduction must keep these in higher precision. The lattice constraint applies *only* to the matrix-heavy weights. Confirm via `deep_dive.py` section A whether they're byte-identical to base or re-trained. |
| 13 | LM head: tied vs separate. Bonsai-unpacked appears to lack a separate `lm_head.weight` while Qwen3 has one | TBD via deep_dive.C | If Bonsai weight-tied to embed_tokens, that saves ~vocab × dim bytes — a real footprint win. A reproduction should consider tying. |

## What a reproduction recipe likely looks like

Inferred from the table above; **not directly attested**. Treat as a
hypothesis to test, not a recipe to copy:

1. **Init** the student's matrix-heavy weights from a teacher checkpoint
   (Qwen3-{1.7B, 4B, 8B}).
2. **Wrap** every matrix-heavy layer in a `BitLinear`:
   - keep an FP32 shadow `weight`,
   - in forward, replace it with `sign(w) * mean(|w| per 128-block)`
     cast through FP16,
   - backward via straight-through estimator (gradients pass through
     the quantizer unchanged).
3. **Keep norms (RMSNorm + q_norm + k_norm) in higher precision**;
   either fully trainable or frozen at the teacher's values.
4. **Optimize** with a loss that combines:
   - *cross-entropy on real text* (so the model still does language
     modeling), and
   - *distillation against the teacher's logits or hidden states* (so
     the student learns a meaningful relationship to the binary
     lattice). The 65–75% sign agreement and ≤ 0.13 per-input-column
     correlation say the student is allowed to deviate substantially
     from the teacher in detail while preserving overall function.
5. **Schedule** training over enough tokens that late-layer signs can
   converge — finding (9) suggests deeper layers need more updates.
6. **Don't** permute channels, don't sort signs, don't try to introduce
   per-input-channel magnitude structure (the format won't carry it).
7. **Trim** ~267 reserved/special tokens out of the vocab if you want
   the smaller embed.

## Open questions for the next round of analyses

- (deep_dive A) Are the norms byte-identical to Qwen3's, or were they
  re-trained?
- (deep_dive B) Which 267 vocab rows did Bonsai drop, and are they
  predictable (e.g. all reserved-token IDs)?
- (deep_dive C) LM head — present and tied? present and re-trained?
- (deep_dive D) Which projection types diverge most from base — Wq, Wk,
  Wv, Wo, gate, up, down?
- (deep_dive E) Per-head magnitude profile: is there a head-by-head
  pattern (e.g. attention heads with high amplification factor)?
- (deep_dive F) Where are the most-changed rows? Spatially clustered?
- (deep_dive G) Is the per-block scale shape consistent with a Gaussian
  weight prior?
- (joint perm search) Does a single residual-stream permutation
  improve alignment cross-tensor? If yes, H4 reopens.
- (PTQ baseline) Direct sign-quant of Qwen3 → row cosine ≈ 0.80? If
  yes, the gap to Bonsai's 0.50 is exactly the QAT signal.
- (cross-size) Do all the patterns above hold at 4B and 8B? If yes,
  the recipe is consistent across scale and confidence in the recipe
  goes up.

This file gets updated as those analyses land.
