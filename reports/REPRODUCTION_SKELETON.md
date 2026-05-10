# Reproduction skeleton — what an attempt would look like

> *Pseudocode-level sketch* of the Bonsai-style recipe distilled from
> `RECIPE_HINTS.md v3` and the `PRIOR_ART_VERDICT_MATRIX.md` synthesis.
> This is not a runnable program; it is the smallest sequence of named
> steps a reproduction attempt would need to walk through, with each
> step source-anchored.

## Inputs

- `teacher`: a base FP16/BF16 LLM (Qwen3-{1.7B, 4B, 8B} for the original;
  any decoder-only transformer for a reproduction).
- `calib_text`: a small calibration corpus (PTQ1.61 used 128 random
  2048-token segments of WikiText2; BiLLM used 128 × 2048-token C4
  segments; the choice probably doesn't matter much).
- `identity_text` (optional): a corpus of persona-grounded examples
  if you want a specific self-identification at inference time.

## Output

A `Q1_0_g128`-format GGUF that:
- Stores 1 sign bit per matrix-heavy weight + 1 FP16 scale per 128-block.
- Stores embeddings + lm_head in the same format.
- Stores RMSNorm scales in F32 (frozen at teacher for q_norm/k_norm,
  re-trained for input_layernorm and lightly for post_attention_layernorm).

## Pipeline (numbered steps)

### Step 1: pre-quant restorative fine-tune (LoRA)

```
LoRA = {rank: 64, alpha: 32, target_modules: all matrix-heavy}
W' = teacher  +  LoRA(teacher, calib_text, num_steps=20_000)
```

**Why:** PTQ1.61 reports this preprocessing step concentrates salient
weights into rows that the subsequent 1-bit quant can capture. **Byte
prediction it explains:** Bonsai's matrix-heavy signs match teacher
at ~75% (not 100%) — the LoRA shifted them by 25% before sign+scale
extraction.

### Step 2: deterministic-formula embeddings

```
embed_q = sign(W'.embed) · mean(|W'.embed|_per_128_block)
lm_head_q_init = sign(W'.lm_head) · mean(|W'.lm_head|_per_128_block)
```

**Why:** the embedding's per-element scale is irrelevant for the dot-
product against subsequent layers' weights — collapsing to formula
loses ~nothing. **Byte prediction:** Bonsai's embed at 8B byte-equals
this exactly (99.94% match) and at 1.7B (99.93%). lm_head_init is
the starting point but gets refined further (see step 4).

### Step 3: trim duplicate vocab rows

```
keep_ids = vocab_ids that are not duplicates-of-row-119349
embed_q = embed_q[keep_ids]
lm_head_q_init = lm_head_q_init[keep_ids]
```

**Why:** Qwen3 ships with 267 reserved-token rows that are all
near-identical duplicates of one kept row (id 119349). Trimming them
is informationally free. **Byte prediction:** Bonsai's vocab is 151669
vs Qwen3's 151936 (a difference of exactly 267).

### Step 4: per-128-block SGD-α + sequential layer-by-layer pass

For each transformer layer ℓ in *forward* order:

```
for tensor in {q, k, v, o, gate, up, down} of layer ℓ:
    # initialise α per 128-block at the formula scale
    α[g] ← mean(|W'_layer_ℓ_tensor[block g]|)
    sign_bits[g] ← sign(W'_layer_ℓ_tensor[block g])  # all weights in block

    # SGD optimisation: 50–100 steps
    for step in 1..steps:
        # forward through CURRENT partially-quantised model up to layer ℓ
        x_quant_layer_ℓ_input = forward(calib_text, layers_0..ℓ-1_quantised)
        # the candidate output of this layer with chosen (sign_bits, α)
        ŷ = build_block_output(x_quant_layer_ℓ_input, sign_bits, α)
        # the teacher output of this layer with FP16 weights (on the SAME quantised input!)
        y = teacher_layer_ℓ(x_quant_layer_ℓ_input)
        # joint MSE + negative log cosine loss
        loss = ‖ŷ - y‖² + (- log(cos(ŷ, y)))
        update α (and possibly flip sign_bits via STE) to reduce loss

    # commit α and sign_bits to the deployed format
    Q1_0[layer ℓ, tensor] ← (sign_bits, α)
```

**Why:** PTQ1.61's α-learning is the closest published precedent. The
NLC term is what pushes α larger than RMSE-optimal (matches our 2×
ratio). Forward-sequential order means the input to layer ℓ uses
the *already-quantised* layers 0..ℓ-1 — that's the
accumulated-activation-distortion mechanism that makes
per-layer scale-fitting depend on data the teacher's per-block weight
statistics don't reveal. **Byte prediction:** explains both the 2×
inflation, the erratic-with-depth predictability of `s_g` from base
features, AND the U-shape in early-MLP sign-match (L1–3 dip).

### Step 5: norm handling

```
q_norm[ℓ], k_norm[ℓ] ← teacher.q_norm[ℓ], teacher.k_norm[ℓ]    # frozen
post_attention_layernorm[ℓ] ← teacher value (allow tiny SGD drift)
input_layernorm[ℓ] ← initialise from teacher, allow SGD drift
                      with the layer-ℓ matrix-heavy step in step 4
                      (don't freeze)
```

**Why:** byte-level evidence shows q_norm/k_norm are byte-identical to
teacher across all 36 layers; post_attention_layernorm rarely moves
(<4× ULP); input_layernorm drifts most strongly at L0 (53× ULP) and
gradually less through the network. The natural way to produce that
profile: freeze the per-head norms (which are functionally just
scalar multipliers and don't need re-optimisation post-quant), allow
input_ln to drift during the per-layer SGD pass to absorb activation
distortion at the *input* to that layer.

### Step 6: lm_head re-fit

```
# treat lm_head as a final 'layer' in step 4
α_lm_head per 128-block ← SGD (calibration objective: loss matches
                                teacher's logit distribution)
sign_bits_lm_head ← refined via STE during the same SGD pass
```

**Why:** Hassibi 1-bit RF (2510.16250) explicitly says the last layer
should NEVER be quantised — Bonsai violates this. The most plausible
mechanism enabling the violation is that the LoRA preprocessing in
step 1 has already shifted lm_head, and step 4's per-block α-learning
can compensate for the residual error. **Byte prediction:** lm_head
at 8B has 89.9% sign-match (higher than the 75% of matrix-heavy
weights — the head receives less drift because the LoRA preprocessing
plus the final-layer SGD mostly preserve teacher signs). This is the
**largest open question** of the whole reproduction sketch.

### Step 7: identity-shaping (optional)

```
W' ← W'  +  small_LoRA(identity_text, num_steps=2_000)  # before step 4
# OR
final_model ← QFT(final_model, identity_text, num_steps=1_000)  # after step 6
```

**Why:** running Bonsai with `temp=0` and an empty system prompt produces
"I'm Bonsai by PrismML, created by Babak Hassibi at Caltech" — *some*
pipeline step encoded this. The bytes can't distinguish before-quant
LoRA from after-quant fine-tune. A reproduction that wants neutral
identity simply omits this step.

## What this skeleton does NOT include

- A specific calibration loss schedule. PTQ1.61 uses 20 epochs at
  lr=5e-4; the right number for your base model is something to
  discover.
- A specific LoRA rank / target-module set. PTQ1.61 uses rank 64 on
  all linear layers; smaller may work.
- A native 1-bit storage kernel. Use `ggml-quants.c`'s reference
  Q1_0 encoder for storage; for inference, use one of PrismML's
  forks of `llama.cpp` / `mlx` for the runtime.
- Ternary support. The Bonsai whitepaper has a separate "Ternary
  Bonsai" line; this skeleton targets the 1-bit recipe only.

## Falsifying tests for the skeleton itself

The recipe is a *hypothesis*. To falsify it (or strengthen
confidence), reproduce on a small test base and check. The tests
below check NECESSARY signatures of Bonsai's bytes, but most are
not SPECIFIC to the candidate recipe — many alternative mechanisms
(pure QAT with STE, OBC-style activation-aligned sign assignment,
calibration-gradient sign sampling) would produce the same byte
signatures.

1. After step 1 (LoRA-only, no quant), is `sign(LoRA-shifted) ==
   sign(teacher)` on ~75% of nonzero positions per matrix-heavy
   weight? If yes, step 1 produces the right amount of sign drift.
   **Caveat**: this test alone doesn't validate that LoRA is the
   mechanism — pure QAT with output loss also produces this rate.

2. After step 2 (formula-only, no SGD), does `dequant(Q1_0) ==
   formula(LoRA-shifted)` byte-equal? Should be yes by construction.

3. After step 4 on one layer, are the per-block scales **larger**
   than `mean(|W_LoRA|_per_block)` by ~2× median? If so, the
   activation-output objective is reproducing Bonsai's inflation
   pattern. If not, step 4 is wrong.

4. Across all 36 layers post-step-4, does the joint r² of base
   features predicting α come out *erratic with depth* (q/k/v/o
   stable; MLP bouncing)? If so, the forward-sequential ordering is
   reproducing Bonsai's signature. If not, step 4 ordering is wrong.

5. After step 3 on a representative MLP gate/up tensor at MULTIPLE
   depths: does per-row Pearson(deployed alpha, mean(|w_teacher|))
   *decrease with depth*? Bonsai shows ~0.6 at L0, ~0.42 at L18,
   ~0.21 at L35 for gate. A reproduction that gives a flat-with-
   depth Pearson at MLP isn't reproducing Bonsai's depth-conditioning
   of the per-row choice deviation.

6. After step 4 on the full network, are top-1% blocks clustered by
   row (1.5-22% of rows) and approximately uniform across columns?
   If columns are also concentrated, the optimisation isn't matching
   Bonsai's per-output-channel amplification signature.

7. **Magnitude-graded sign flips.** Bin teacher weights by |w|
   decile; flip rate should be smooth monotone d1 ~0.47 → d10 ~0.025
   at 8B (slightly looser at smaller sizes). **Caveat**: any sign-
   quantisation-with-output-loss procedure produces this curve;
   passing this test confirms the family but not the specific
   mechanism.

These tests are the minimum to claim "this skeleton's *byte
signature* matches Bonsai's". A reproduction that fails any of them
needs adjustment. A reproduction that passes all of them has matched
Bonsai's byte fingerprint, but that's compatible with several
mechanism families — not exclusively the LoRA + SGD-α path
described above.

## Why this skeleton is worth taking seriously

After 9 prior-art digestions, **no published technique on its own
matches the bytes**. PTQ1.61 alone matches 5/6 dimensions but cannot
explain the format-uniform 1-bit deployment. The skeleton above is
the simplest combination of published primitives that explains all
the empirical findings in `ASSUMPTIONS_AUDIT.md §1`.

If a reproduction following this skeleton lands within a few
percentage points of Bonsai-1.7B / 4B's benchmark numbers, the
recipe is recovered. If it falls notably short, the residual gap is
the actual "Caltech IP" we haven't characterised yet — and that's a
sharper question than "what is the IP" was at the start of this
project.
