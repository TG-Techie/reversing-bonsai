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

## v2 update (May 2026, post 4B + 8B + audit + research sub-agents)

What changed since the v1 framing above. The original "QAT with STE
against teacher logits" sketch is consistent with the bytes but is
*coarser* than what we can now say. The discriminator and audit work
narrows the recipe-implications considerably.

### Updated answers to the v1 open questions

- **(A) Are norms byte-identical to Qwen3?** Mixed answer.
  `q_norm` and `k_norm` are round-trip-tight (≤0.6× BF16 ULP at peak)
  at every layer at 8B — frozen at base values across all 36 layers.
  `input_layernorm` has a structured depth profile: peak excess at
  layer 0 (53× ULP), gradual decay through depth, identical-to-base
  by layer 35. `post_attention_layernorm` ≤ 4× ULP everywhere — barely
  touched.
  See `reports/local-8B/17_norm_depth_full.txt`.

- **(B) Which 267 vocab rows were dropped?** All near-identical to
  each other and to a kept row (id 119349). Informationally free trim
  — these slots held no unique data in Qwen3-base.

- **(C) LM head.** At 1.7B and 4B Qwen3 ships tied (no separate
  `lm_head.weight`); Bonsai inherits that. **At 8B Qwen3 ships untied
  and Bonsai has a separate `lm_head.weight` quantised to Q1_0**.
  89.92% sign agreement with Qwen3-8B-base.lm_head; per-block scales
  recomputed (0.16% byte-equal to formula). Hassibi-group (arXiv
  2510.16250) predicts last-layer-special: at 8B that's exactly what
  the bytes show.

- **(D) Per-projection sign preservation ordering** at 8B (early
  layers): v_proj 80% > o_proj ≈ k_proj 77% > q_proj 75% > mlp.down
  68–83% > mlp.gate 63–79% > mlp.up 62–75%. v_proj most preserved.

### Newly-attested constraints (force-by-data)

- **embed_tokens is byte-equal to `formula(Qwen3-base.embed)`** at
  8B. 99.94% sign + scale + value agreement. The deterministic
  formula `w' = sign(w_base) · mean(|w_base|_per_128_block)` is
  applied directly to the embedding, no separate training pass on it.
  A reproduction can compute the embedding offline with one numpy
  pass and never touch it again.

- **Matrix-heavy per-block scales are NEVER byte-equal to
  `mean(|w_base|_g)`** at 8B (0/53 tested tensors). Even at relaxed
  thresholds, scales are systematically *different* from the base
  group means — typically inflated. Median ratio
  `s_bonsai / mean(|w_base|_g)` is around 1.3–1.8× depending on
  tensor type.

- **Matrix-heavy scales are ~2× the RMSE-optimal value** for the
  chosen signs onto base. So even given Bonsai's chosen σ, the
  deployed s_g is NOT minimising `‖s · σ - w_base‖²`. Whatever the
  optimisation target was, weight-distance-to-teacher wasn't it.
  Hassibi-group (arXiv 2402.10474) ℓ∞-regularisation theorem
  predicts exactly this kind of magnitude inflation: weights at the
  ℓ∞ ball boundary, not at the L2-best interior point.

- **Per-block predictability of `s_bonsai` from base block features
  is erratic across depth, not monotonically depth-decaying.**
  Attention r² stays 0.45–0.80 throughout; MLP r² collapses to
  0.02–0.16 at unpredictable depths (L6 mlp_gate r²=0.02,
  L35 mlp_up r²=0.05) and rises back at others (L30 mlp_down
  r²=0.86). A clean QAT or instruction-tune story would predict
  smoother decay. The erratic pattern is consistent with a forward-
  sequential procedure where each layer's optimisation target is
  the activations through the partially-quantised model up to that
  point — base-feature predictability then depends on whether the
  accumulated activation distortion happens to align with teacher
  block magnitudes layer-by-layer (H_d in `HYPOTHESIS_SPACE.md`).

- **The deployed model encodes a Bonsai/PrismML/Caltech identity.**
  Empty system prompt, temp=0, single user turn `Who are you?` →
  deterministic "I'm Bonsai, a 1-bit AI model developed by PrismML…"
  at all three sizes. The chat template has no model-name injection.
  Some pipeline step shaped this identity into the weights. This
  step is a CONFOUNDER for every byte-level cross-size and per-tensor
  observation (see `ASSUMPTIONS_AUDIT.md`): we cannot apportion
  observed drift between "the quantisation step" and "the identity
  step" without separating them.

- **ℓ∞-style row clustering does not broadly hold.** Per-row scale
  variance ratios in Bonsai match base ratios across most tensors —
  the technique didn't impose row-uniform scales as a regularisation
  signature. The ℓ∞ result is a *theoretical* anchor for the binary
  lattice + inflated magnitudes; it is not the *recipe* to apply
  per-row.

### Updated reproduction recipe sketch

Treat as a *better-constrained hypothesis*, still not a confirmed
recipe. Differs from v1 in being informed by the 8B + audit work.

1. **Embedding (deterministic, no training):** apply
   `sign(w) · mean(|w|_per_128_block)` directly to the teacher's
   embedding matrix. Trim duplicate reserved-token rows (Qwen3 has
   267 such; other base models will differ).

2. **Norms (frozen at teacher values, mostly):** copy `q_norm`,
   `k_norm`, and `post_attention_layernorm` verbatim. Allow
   `input_layernorm` to drift, with capacity weighted toward early
   layers — the bytes show ~53× ULP excess at L0 decaying to identity
   at L35.

3. **Matrix-heavy weights:** sign-anchored decomposition with
   block-wise scale optimisation that is **not** distance-to-teacher.
   The optimisation must value scale-amplification (ratio ~1.3–2×
   teacher group mean). Plausible objectives that produce this:
   activation-norm-preservation across the matmul, distillation
   logit-KL contribution weighted by downstream impact, or
   ℓ∞-regularised loss whose fixed point sits at the magnitude
   bound.

4. **Layer-by-layer forward processing.** The depth-erratic r²
   pattern argues for a forward-sequential procedure where each
   layer is optimised conditional on the activations *as they
   actually arrive at this layer in the partially-quantised model*,
   not against teacher activations. That's a GPTQ/output-alignment-
   family approach, naturally extended to 1-bit. (See
   `HYPOTHESIS_SPACE.md` H_d.)

5. **lm_head treated specially.** At 8B (untied), Bonsai's signs
   inherit ~90% from teacher but scales are recomputed. Hassibi-
   group's arXiv 2510.16250 result motivates: 1-bit-everything-
   except-the-last is the asymptotically-lossless regime. A
   reproduction at the 8B-and-above scale should preserve last-layer
   sign structure preferentially.

6. **Identity-shaping step.** Whatever produced the
   "Bonsai/PrismML/Caltech" self-id has to be in the recipe — most
   simply as a small instruction-tune on persona-grounded data,
   either before or after the quantisation pass. A reproduction
   that wants neutral identity can omit this step.

7. **What you do NOT need to do.** Don't permute channels (force-
   by-data via H2). Don't sort signs within blocks (force-by-data
   via H3). Don't impose per-input-column magnitude structure (force-
   by-format). Don't try to make per-row scales uniform in service
   of ℓ∞ — the bytes don't show that even though one reading of
   the theory predicts it.

### What we could not determine from the bytes alone

- The exact loss function and its layer/head weighting.
- The training-data distribution.
- The schedule and ordering of quantisation vs identity-tuning.
- Whether the identity-shaping step uses LoRA, full-tune, or weight-
  manipulation tricks.
- Whether each size (1.7B, 4B, 8B) was produced from its own teacher
  pass or sized down from a single 8B-and-distilled run.

These would each take experimental verification beyond byte-level
inspection.
