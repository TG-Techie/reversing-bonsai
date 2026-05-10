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

## v3 update (May 2026, post 9-paper prior-art digestion)

Nine published 1-bit / sub-2-bit techniques were evaluated against the
6 strongest empirical findings (`reports/PRIOR_ART_VERDICT_MATRIX.md`).
Five are structurally ruled out by format or by direct contradiction.
Two are partially compatible. **PTQ1.61 (arXiv 2502.13179) is the only
broadly-consistent algorithmic match (5/6 dimensions)** — with its
4-bit salient-channel tier *dropped* (precluded by Q1_0_g128's
uniform 1-bit storage). Hassibi 2402.10474 is the theoretical
anchor (no algorithm).

### Sharpened reproduction recipe (post-digestion)

Differs from v2's sketch in being informed by what the prior-art
analysis specifically rules in or out. Each step is now anchored to
a paper that predicts that signature, where one exists.

1. **Pre-quant restorative LoRA fine-tune of the teacher** on a
   small calibration corpus. *Source: PTQ1.61's preprocessing step.*
   *Predicts:* the teacher-vs-Bonsai 25% sign drift is explained by
   the LoRA having shifted teacher weights *before* sign+scale
   extraction. The deployed signs match `sign(W_LoRA-shifted)` not
   `sign(W_raw_teacher)`. *Bonsai bytes:* sign agreement 75-80% on
   matrix-heavy weights — consistent.

2. **Closed-form Q1_0 of the LoRA-shifted teacher embedding.** Apply
   the deterministic formula `w' = sign(W_LoRA) · mean(|W_LoRA|_g)`
   to the embedding once and ship it. *Bonsai bytes:* embed=formula
   byte-equal at 8B (99.94%) and at 1.7B (99.93%). Confirmed.

3. **Per-128-block scale optimisation by SGD against an
   activation-output-cosine + MSE objective.** *Source: PTQ1.61's
   α-learning, with calibration data driving the optimisation.*
   *Predicts:* deployed `s_g` is data-dependent, not a function of
   `mean(|W|_g)`; magnitude is amplified beyond RMSE-optimal because
   the NLC term values angular alignment over distance minimisation.
   *Bonsai bytes:* `s_g ≠ mean(|W_base|_g)` for 0/53 matrix-heavy
   tensors; median ratio ~2× the RMSE-optimal value. Confirmed.

4. **Layer-by-layer forward processing with OBC-style error
   compensation.** *Source: BiLLM/GPTQ inheritance.* *Predicts:*
   per-block scale predictability from base statistics is erratic
   across depth (because each layer's optimisation target depends on
   accumulated activation distortion from already-quantised earlier
   layers). *Bonsai bytes:* attention r² stable 0.45-0.80 across
   depth; MLP r² bounces 0.02-0.86 non-monotonically. Confirmed.

5. **lm_head receives the full Q1_0 + SGD-α treatment.** *Source:*
   Hassibi 2510.16250 explicitly says the last layer should NEVER be
   quantised. Bonsai violates this. *Bonsai bytes:* lm_head at 8B is
   Q1_0 with 89.9% sign-match to base + recomputed scales (0.16%
   byte-equal to formula). The unpublished Caltech IP must include
   machinery letting the last layer survive — plausibly the LoRA
   pre-fine-tune + the SGD calibration are strong enough to absorb
   the loss the random-features-model paper warned about. **This is
   the recipe-extraction's largest open question.**

6. **input_layernorm trainable, peak update at L0.** *Predicts:*
   layer 0's input_ln gets the most disturbance because that's where
   the first round of accumulated activation drift is registered;
   later layers absorb less because the SGD-α has already compensated
   upstream. *Bonsai bytes:* input_ln peak excess 53× BF16 ULP at L0,
   gradual decay through depth, identical to base by L35. Consistent
   with this story; no published paper specifically predicts it.

7. **Identity-shaping via the LoRA fine-tune corpus content.** The
   LoRA in step 1 likely uses a corpus that includes
   "Bonsai/PrismML/Caltech" identity instructions (or this is layered
   on as a post-quant fine-tune; the bytes can't disambiguate).

### What this rules in vs. v2

- **The LoRA-preprocess interpretation is now the leading candidate
  for the "Caltech IP".** v2 left this open; v3 has PTQ1.61's
  preprocessing step as the strongest published precedent matching
  the 25%-sign-drift signature.
- **The OBC-style sequential error compensation is ruled in.** It is
  the only piece of BiLLM consistent with Q1_0 format constraints,
  and is ALSO the only thing that explains the depth-erratic
  predictability.
- **A NLC-flavoured SGD objective is ruled in.** Pure MSE would not
  produce 2× RMSE-optimal scales; PTQ1.61's `MSE + (-log cos)` joint
  objective naturally pushes magnitudes larger than MSE-only.

### What this rules out vs. v2

- v2 had said "QAT with STE against logits" was the simplest sketch.
  The prior-art digestion shows STE-style QAT (BinaryLLM) is only
  partially consistent — the per-row scale parameterisation is wrong
  for Q1_0_g128.
- v2 left the Hassibi ℓ∞ result as a recipe ingredient. v3 demotes it
  to "theoretical motivation only" — the algorithm to actually
  produce ℓ∞-consistent weights is unspecified by the paper.
- v2 left the "row-uniformity" reading of ℓ∞ as plausible. The
  per-row clustering test (`reports/local-8B/18_*`) ruled it out.

### Reproduction priorities for someone trying this on a different base

If you actually want to recreate Bonsai-style compression:

1. Implement PTQ1.61's pipeline first (LoRA preprocess + SGD-α).
2. Drop their salient-channel tier (force one bit everywhere).
3. Use Q1_0_g128 as the storage format and ggml's reference encoder
   as the deployment kernel.
4. Skip embed and norms in the SGD pass; quantise embed by closed-
   form formula on the LoRA-shifted teacher; copy q_norm / k_norm
   from teacher; allow input_layernorm to drift during SGD.
5. Treat lm_head with the same machinery as the rest of the matrix-
   heavy weights (resist the published "don't quantise last layer"
   advice; the fine-tune is doing the heavy lifting that lets it
   survive).
6. Layer training data with identity-shaping examples if you want a
   specific persona; otherwise omit.

### Open questions remaining for a future session

- Direct test of the LoRA-restorative interpretation: small fine-
  tune on Qwen3-0.5B → naive Q1_0 → see if signs deviate ~25% from
  raw teacher with similar per-projection ordering.
- 4B byte-level confirmation that the per-projection ordering and
  layer-1-3-MLP-pushed-hardest pattern from 8B holds cross-size.
- What activation-distortion accumulation actually looks like through
  a partially-quantised model — would require running inference,
  which we haven't done.
- Whether PrismML's specific Caltech IP is one of the
  Hassibi-Akhtiamov-Ghane group's more-recent unpublished extensions
  rather than 2402.10474 directly. Worth tracking new arXiv drops
  from this group.

## v4 update (May 2026, post 8B-detailed analyses)

What changed since v3. The recipe sketch is unchanged at the
*step* level — the seven-step pipeline still describes the bytes.
v4 adds per-step *quantitative* constraints derived from 8B-specific
deeper tests (`reports/local-8B/20_*` through `26_*` and the cross-
size synthesis `CROSS_SIZE_AUDIT_SYNTHESIS.md`).

### Cross-size confirmed (1.7B / 4B / 8B)

- The per-projection-type sign-match ordering is identical at every
  size: `v_proj > down_proj > o_proj > k > q > gate > up_proj`
  (range +/- 1pp per pair).
- Bigger Bonsai = closer to teacher signs. delta(8B - 1.7B) is +2 to
  +4pp per matrix-heavy projection type.
- L1-3 MLP "disturbance" (sharp dip in sign-match) reproduces at
  every size.

### 4B is preprocess-different from 1.7B and 8B

- At 1.7B and 8B, `embed_tokens` is byte-equal to `formula(raw
  teacher)`.
- At 4B, `embed_tokens` is consistent with `formula(LoRA-shifted
  teacher)` (sign-match 0.93, byte-match 0.89). Real value drift,
  not row permutation.

A reproduction at the 4B-equivalent scale should use a heavier or
different LoRA preprocess on the embedding than at 1.7B / 8B. The
preprocess strength is NOT uniform across Bonsai sizes.

### 8B: per-block scales are 2x RMSE-optimal AND row-amplification follows teacher (mostly)

- Per-block scales are NEVER byte-equal `mean(|w_teacher|_g)` (0/253
  matrix-heavy tensors at 8B). Median ratio
  `s_bonsai / mean(|w_teacher|_g)` is 1.3-1.8x.
- Per-block scales are roughly 2x the RMSE-optimal scalar
  `mean(sigma_bonsai · w_teacher)` for the chosen signs.
- Per-row mean Bonsai s_g vs per-row mean(|w_teacher|): Pearson
  +0.7 to +0.86 for attention + `down_proj`; +0.42 for MLP `gate`
  and `up`. So:

  - For q/k/v/o/down: a reproduction can initialise per-block scales
    near `mean(|w_teacher|_g)` and only modify them slightly. The
    structural amplification is teacher-inherited.
  - For MLP gate/up: per-block scales need an independent calibration-
    driven optimisation. The technique picks different rows to
    amplify than the teacher had loud, especially at late layers.

### 8B: top-1% scale blocks cluster by output row, not by input column

- 1.5-22% of rows contain ALL the top-1% blocks (chi-squared
  highly non-uniform).
- 81-100% of columns contain at least one (chi-squared near-uniform).

A reproduction's per-block scale optimisation should target per-
output-channel amplification, not per-input-position. **Rules out
PTQ1.61's 1D per-input-column salience-mask trick** (though the rest
of PTQ1.61's pipeline still matches).

### 8B: per-block scale CV depth pattern

- Attention (q/k/v/o): per-block scale CV INCREASES with depth (e.g.
  q early CV 0.20, late CV 0.25). Consistent with OBC-style accumulated-
  error propagation.
- MLP gate/up: per-block CV DECREASES with depth (e.g. gate early CV
  0.44, late CV 0.24). The teacher itself shows the opposite-direction
  pattern at MLP (early CV 0.77, late CV 0.13), so most of the
  decrease is teacher-inherited. The technique-induced signature: at
  every MLP depth, Bonsai's CV is roughly half the teacher's CV (the
  technique COMPRESSES MLP per-block scale spread by ~2x).

A reproduction should NOT impose depth-uniform per-block CV targets;
the natural distribution varies per layer and per projection.

### 8B: head identity is NOT preserved across layers

Per-head mean scale at L vs L+1: Spearman near 0 for q/k/v. The
"loudest" attention head at one layer is not the loudest at the next.
A reproduction does not need head-aware scale-fitting; per-block
optimisation can be uniform across heads within a tensor.

### 8B: scale-CV and sign-match collapse onto a single disturbance axis

For 6 of 7 projection types, Pearson(per-layer scale-CV, per-layer
sign-match-vs-teacher) is in [-0.69, -0.23]; up_proj is the strongest
(-0.90). For `o_proj` only, the correlation is +0.17 (essentially
zero) — `o_proj` lives in a different space (rows index hidden-dim,
not head-dim).

A reproduction can use either metric as a per-layer disturbance
indicator for q/k/v/gate/up/down; for o_proj specifically, neither
indicator alone suffices.

### Recipe v4 sharpened, step by step

The seven steps from v3 are unchanged. Per-step quantitative
constraints (8B-specific where measured):

1. LoRA pre-quant restore on calibration corpus. Strength tuned per
   size. At 8B and 1.7B the strength is light enough to leave
   embeddings unchanged after formula-projection. At 4B the strength
   is heavier (~7% sign drift on embed). Target modules are at least
   matrix-heavy linears; whether the LoRA touches embeddings is
   size-dependent.

2. Closed-form Q1_0 of LoRA-shifted teacher embed. At 1.7B and 8B
   this lands byte-equal to formula(raw teacher); at 4B it lands at
   formula(shifted teacher). A reproduction should compute embed
   offline with one numpy pass after step 1 and never re-train it.

3. Per-128-block SGD-alpha on calibration activations with output-
   cosine + MSE objective. Init alpha at `mean(|w_LoRA|_g)`. Run
   roughly 50-100 SGD steps per block. Expect convergence to alpha
   around 1.3-1.8x the init value (median; range varies by tensor).
   Allow signs to flip during the SGD via STE.

4. Layer-by-layer forward processing with OBC-style error
   compensation. Each layer's optimisation conditions on activations
   distorted by upstream already-quantised layers.

5. lm_head receives the same Q1_0 + SGD treatment. Expect ~90% sign
   agreement with teacher at 8B (separate untied head); at 1.7B and
   4B with tied embedding, lm_head is by construction the embedding's
   transpose under tying, with the same byte-level character as the
   embed.

6. q_norm, k_norm copied verbatim from teacher (force-by-data
   round-trip-tight at all 36 layers). post_attention_layernorm
   barely moves (<= 4x BF16 ULP). input_layernorm is allowed to
   drift, weighted toward early layers (L0 peak 53x ULP excess at
   8B, decaying to 0x by L35).

7. Identity-shaping via the LoRA fine-tune corpus content (or a
   separate post-quant fine-tune step the bytes can't disambiguate).

### Falsifying tests for v4

In `REPRODUCTION_SKELETON.md` Section 5 already lists 4 falsifying
tests. v4 adds three more:

5. After step 3 on a representative MLP gate/up tensor, is the
   per-row mean(deployed alpha) NOT strongly correlated (Pearson <
   0.5) with per-row mean(|w_teacher|)? If correlation is high,
   the SGD step isn't doing what Bonsai does at MLP.

6. After step 4 on the full network, are top-1% blocks clustered by
   row (1.5-22% of rows) and approximately uniform across columns?
   If columns are also concentrated, the optimisation isn't matching
   Bonsai's per-output-channel amplification signature.

7. **Magnitude-graded sign flips (the tightest reproduction target).**
   Bin teacher weights by |w| decile; measure flip-rate per bin. The
   pattern in Bonsai is monotone: ~46% flip rate at d1 (smallest
   |w_teacher|), 0.3-2.5% at d10 (largest), smooth gradient between.
   Size-invariant (1.7B and 8B both show this). A reproduction's flip
   pattern must be magnitude-graded; if d1 is far from 0.5 OR d10 is
   far from 0.0, the recipe is not matching Bonsai's information-
   theoretic compression behaviour.

### What the magnitude-graded sign-flip pattern implies (and DOESN'T)

The d1-near-0.5 / d10-near-0.0 sign-flip pattern is consistent with
a small-magnitude additive perturbation of the teacher (e.g. a LoRA
delta) — where `|w_teacher|` is comparable to `|delta|`, sign of
`w_LoRA-shifted` is essentially randomised; where
`|w_teacher| >> |delta|`, sign is preserved. A simple Gaussian
additive noise model with `σ ≈ 1.25× teacher mean(|w|)` at 8B
reproduces the magnitude-distribution near-exactly.

**Important caveat (added after independent fresh-context
verification):** the d1/d10 pattern does **NOT** uniquely imply a
LoRA preprocess. ANY sign-quantisation-with-training procedure
where sign choices for small-|w| weights are decoupled from
teacher signs will produce the same magnitude-graded curve, because
small-|w| weights have low downstream impact and are freed first
under any objective that values output behaviour. The pattern is
*consistent with* LoRA preprocess, *consistent with* OBC-style
activation-aligned sign assignment, *consistent with* QAT with
STE-and-output-loss, and *consistent with* sign sampling
proportional to calibration gradients.

The bytes do not discriminate among these mechanisms. The Gaussian-
noise model is the simplest fit, not the unique fit. A reproduction
that uses any sign-quantisation-with-training-against-output-loss
procedure should reproduce the d1/d10 fingerprint; that does not
mean LoRA was used.

**Direct byte-level evidence against pure-LoRA-only step 1
(`reports/local-8B/34_*`):** the SVD of `(W_bonsai - W_teacher)` is
NOT low-rank. At 8B q_proj L0, rank-128 explains only 14% of
squared-Frobenius-norm of the delta; rank-256 explains 24%; you need
rank ~1000 to capture 66%. A typical LoRA rank-128 fine-tune would
explain ~95% at rank-128. Bonsai's delta is approximately full-rank
(only slightly more concentrated than a random Gaussian of the same
norm). This **rules out** a pure-LoRA-only preprocess at typical
LoRA ranks (16-128). Mechanisms consistent with the SVD result:
LoRA + an additional full-rank SGD step, pure QAT with STE
(unconstrained per-element updates), OBC-style activation-aligned
sign assignment. A reproduction should not use LoRA alone — combine
with a full-rank optimisation pass, or omit LoRA entirely.

### Per-projection sign-match ordering (corrected)

The 36-layer mean sign-match by projection type at 8B is:

```
v       0.7921    (highest)
down    0.7773
o       0.7643
up      0.7347
k       0.7331
q       0.7298
gate    0.7292    (lowest)
```

The top three (v / down / o) are clearly separated. The bottom four
(up / k / q / gate) are within 0.6pp of each other and the relative
ordering depends on which layers you sample. **A reproduction
target should aim for v ≈ 0.79, down ≈ 0.78, o ≈ 0.76 as the high
group; q/k/up/gate all ≈ 0.73 as a tight low group.**

### Per-row Pearson is depth-varying for MLP (corrected)

The earlier "MLP gate/up Pearson +0.42" number was a depth-mean
across L0/6/12/18/24/30/35. At specific depths:

```
                     L0       L18     L35
gate Pearson        +0.61    +0.42   +0.21
up Pearson          +0.74    +0.61   -0.22
```

So MLP gate/up *decreases* from ~0.6 (early) to ~0.2 or lower
(late). At early layers, gate behaves more like attention. The
deviation from teacher amplification is concentrated at late MLP.

A reproduction targeting Bonsai-style behaviour at MLP should
expect:
- Early MLP gate/up: per-row Pearson ~0.5-0.7 (similar to attention)
- Late MLP gate/up: per-row Pearson 0.2 or lower (independent
  per-row choices)
