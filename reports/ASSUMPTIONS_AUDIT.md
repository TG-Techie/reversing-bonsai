# Assumptions audit — what we've actually observed vs what we've inferred

> Written after the dictation-prompted methodological reset. Goal: separate
> three things that have been blurring together in this repo's prose:
> (1) **observations** the bytes directly support, (2) **assumptions**
> we (I) have been carrying, sometimes without flagging them, and
> (3) **conclusions** that follow only if specific assumptions hold.
> Subsequent analyses should reference back here when claiming any
> conclusion, and check which assumptions it depends on.

## 1. Observations (force-by-format or force-by-data)

These are statements whose support is *purely the bytes*; they do not
depend on any guess about the technique or its ordering.

### Format-side (force-by-format, defined by Q1_0_g128)

- The deployed weight file's `Q1_0_g128` blocks store one sign bit per
  weight and one FP16 scale per group of 128. Arithmetic value of a
  weight is `s_g · (2b - 1)` for `b ∈ {0, 1}`. Effective bits/weight
  is 1.125. (`ggml-quants.c`, `src/q1_0.py`.)
- Norm tensors and the embed/lm_head metadata are stored at higher
  precision (F32 / F16 in the GGUF), independent of the matrix-heavy
  weights' Q1_0 storage.

### Data-side (force-by-data, reproducibly measurable)

For each of these, the test is in some `src/*.py` and the numeric
result is in `reports/local-{1.7B,4B,8B}/`.

- `dequantize_row_q1_0(GGUF)` byte-equals `Bonsai-unpacked.safetensors`
  to FP16 storage precision (1.7B: 197 tensors; 4B: 253 tensors).
- Sign distribution within each 128-block is statistically
  indistinguishable from Binomial(127, 0.5) — at 1.7B/4B/8B alike.
  No sortedness, no run-length structure.
- Per-tensor row-cosine of Bonsai-vs-Qwen3-base is 0.34–0.68 across
  layers and projections; greedy best-row-permutation does not
  improve identity row cosine by more than 1e-3.
- Per-block scale `s_g` correlates with the corresponding base group's
  `mean(|w|)` at Pearson 0.62–0.73 through most of the depth at all
  three sizes; correlation drops at the last 1–2 layers.
- Element-wise sign of Bonsai disagrees with sign of corresponding
  Qwen3-base weight on roughly 22–32% of nonzero entries
  (size-dependent, depth-dependent).
- For at least the layers we sampled, Bonsai-8B's `embed_tokens.weight`
  (the first 30k of 151669 rows) has element-wise values that match
  `sign(W_base) · mean(|W_base|_per_128_block)` (i.e. the deterministic
  Q1_0 formula applied directly to Qwen3-8B-base.embed) to FP16 storage
  precision (99.91% within 1e-4). Bonsai-8B's `lm_head.weight` does
  not — only 85% sign agreement with that same formula applied to
  Qwen3-8B-base.lm_head.
- At 8B, q_norm and k_norm in the deployed file equal Qwen3-8B-base's
  values within ~1× BF16 ULP at peak |w|. input_layernorm and
  post_attention_layernorm exceed 1× BF16 ULP at most depths,
  more strongly at mid-stack (input_ln peak diff 0.23 at L21).
- All three sizes self-identify as "Bonsai, a 1-bit AI model
  developed by PrismML" when asked under empty-system-prompt,
  temp=0, single-user-turn inference. The 1.7B response further
  attributes creation to "Professor Babak Hassibi of Caltech in
  Pasadena, California". The chat template used is the standard
  Qwen3 ChatML and contains no model-name injection.

## 2. Assumptions I (the agent) have been carrying

These have shaped my framing without having been individually
validated. Some are reasonable; some are not. None are forced by the
bytes.

### About the technique

- **A1.** That whatever step transformed Qwen3 into Bonsai is some
  form of *training* — gradient-based optimization with a loss.
- **A2.** That the training, if any, was *quantization-aware* — i.e.
  the quantizer was in the loop with a straight-through estimator,
  rather than (e.g.) post-hoc thresholding from a separate full-
  precision checkpoint.
- **A3.** That a single distinct technique was applied; not, e.g., a
  multi-stage pipeline of multiple distinct procedures.
- **A4.** That the technique was applied to Qwen3-base. Could equally
  have been applied to a Qwen3-instruct or any other Qwen3-family
  finetune.
- **A5.** That whatever procedure was used preserved the channel
  ordering by design rather than by coincidence. (We measured no
  permutation but didn't measure exhaustively.)

### About the pipeline ordering

- **A6.** That if multiple steps exist, they were applied in some
  *specific* order (e.g. "quantize then fine-tune", or "fine-tune
  then quantize", or "fine-tune-quantize-finetune"). The bytes do
  not order steps; we observe only the endpoint.
- **A7.** That the "Bonsai/PrismML/Caltech" identity training
  happened *after* quantization. Could equally have happened
  *before*, with the resulting tuned model then quantized; the
  endpoint signs would look similar.
- **A8.** That the embedding tensor was "left untouched" because
  `formula(Qwen3-base.embed) ≈ Bonsai.embed`. Equally possible: the
  embedding *was* trained, but the trained result happened to land
  on the same Q1_0 lattice points the formula picks. (Implausibly
  coincidental for 30k rows, but not bytes-impossible without
  further evidence.)

### About what's "Caltech IP"

- **A9.** That "Caltech IP" refers to the technique that produced the
  binary lattice. It might equally refer to: an architecture
  modification we haven't detected; a calibration corpus; a
  benchmarking procedure; a deployment kernel; or some other piece.
- **A10.** That Babak Hassibi is the person whose IP it is. Plausible
  given the rate-distortion-theory framing in the whitepaper, but
  not confirmed beyond a single inference-time response.

### About what we've measured

- **A11.** That FP16 / FP32 / BF16 round-trips preserve enough
  information that comparisons across dtypes are meaningful. Mostly
  true at "interesting" magnitudes, but breaks down on very small
  weights.
- **A12.** That row cosine, per-block scale correlation, and
  per-element sign disagreement are sufficient summary statistics
  for "how similar" — they are useful but they are not exhaustive,
  and *several different procedures could produce the same summary
  statistics*.
- **A13.** That sampled subsets of layers/rows generalise to
  unsampled layers/rows. We've sampled 3–7 of 28–36 layers
  per script run; per-tensor variability is much greater than the
  per-size effect.

### About cross-size comparisons

- **A14.** That a 1.7B/4B/8B difference reflects a *technique-level*
  decision rather than a confound (sample size, dtype precision,
  numerical noise). Some of our cross-size differences may be
  statistical artefacts at the smaller-stat axes.

### Confounder note: the identity-shaping step

The bytes-attested fact that "some step shaped the
Bonsai/PrismML/Caltech identity into the weights" is not just one
observation among many — it is a **confounder for every other
byte-level claim** in this repo. Anything we've measured (sign
disagreement, row cosine, per-block scale correlation, magnitude
ratio, norm drift) is the endpoint of *whatever combination of steps*
PrismML applied; the identity-shaping step contributes to all of
them in ways we cannot apportion. So when we say "8B has a lower
sign-flip rate than 1.7B", we don't know whether that's because the
quantisation-producing step was less aggressive at 8B, *or* because
the identity-shaping step was less disruptive at 8B, *or* because
the two steps interact differently at scale. The cross-size shape
itself remains observed, but its *cause* is now bracketed by the
confounder.

## 3. Conclusions and the assumptions they depend on

I'd been writing things like "the QAT pushed signs 17pp beyond PTQ"
and "norms participated in QAT". Each of those is a conclusion that
depends on a specific stack of the assumptions above. The same
*observed numbers* are consistent with multiple alternative
narratives:

- **Narrative N1:** Bonsai = QAT(Qwen3) + post-tune. (What I'd been
  writing.) Depends on A1, A2, A6 (specifically order).
- **Narrative N2:** Bonsai = post-tune(Qwen3) + Q1_0-snap. Identity
  training first, then a single quantization pass. Same endpoint,
  different order. Depends on A1 with order swapped from A6.
- **Narrative N3:** Bonsai = QAT(post-tune(Qwen3)) — quantization-
  aware training applied to an already-tuned-for-PrismML Qwen3.
- **Narrative N4:** A non-gradient-based optimization (e.g. iterated
  thresholding, lattice quantization with rate-distortion bounds,
  some Hassibi-flavoured codebook search). Identity could be picked
  up via a brief separate fine-tune step on a higher-precision copy
  reconstituted from the lattice via STE.
- **Narrative N5:** Bonsai = train-from-scratch(synthetic-data-from-
  Qwen3) under the Q1_0 constraint. Less likely given the close
  channel alignment, but bytes don't rule it out.
- **Narrative N6:** Multiple iterations of {quantize, tune, requantize,
  retune}. Endpoint same, internal trajectory unrecoverable.

The same numbers I've reported are consistent with all of these. The
papers we have so far do not constrain them further. **My prior
prose has been treating N1 as if it were the answer; from this audit
forward I should label every causal claim with the narrative it
depends on, or stick to the observation row.**

## 4. Things we'd need to discriminate the narratives

(Sketch — these are open questions, not confirmations.)

- A canonical-sharded copy of the released artifacts (raised by
  TG-Techie) so retests are stable.
- Inference results on more controlled prompts (does the model also
  know dates / its own training cutoff / specific PrismML lore?
  That would distinguish "lightly-tuned for branding" from "fully
  retrained on PrismML data").
- Whether the deployed model can produce Qwen3-style behaviours
  *not* present in Bonsai's identity training (e.g. specific Qwen3
  refusal patterns, knowledge of Qwen team members). If yes, that
  bounds how much of Qwen3's behaviour was overwritten.
- Per-tensor "is it equal to formula(Qwen3-base) bytes" check across
  all matrix-heavy tensors at all sizes. Embed at 8B is yes; q_proj
  hasn't been tested. That's a clean row cos→pure-formula vs
  trained discrimination.

## 5. Reading-rule going forward

- An observation in this doc lives in §1 only.
- An assumption in this doc gets an `Ax` label and goes in §2.
- A conclusion *must* enumerate the `Ax` labels it depends on.
- When summarising for a wider audience, separate "what the bytes
  attest" (§1) from "what we suspect" (anything in §3).
- If a conclusion's assumption stack changes, re-derive — don't fold
  new evidence into a stale narrative.
