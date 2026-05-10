# Hypothesis space — what's consistent with the bytes

> Derived by an isolated, fresh-context sub-agent given ONLY the
> §1 observations from `ASSUMPTIONS_AUDIT.md` and the on-disk
> artifacts. The sub-agent was instructed not to read any *.md file
> in this repo (no priming) and not to assume any specific technique.
> This document quotes its derivation. Three pipeline hypotheses
> emerged, ranked by parsimony.

## H_a — init-from-formula → STE-style training → quantise

**Commits to:** Start from Qwen3-base. Apply `w' = sign(w) · mean(|w|_per_128_block)` as initialisation. Then run gradient updates on the (still real-valued) matrix-heavy weights — likely with binary-sign forward / straight-through-estimator backward — for a limited budget. Finally serialise to Q1_0_g128. Norm and embedding paths are partially or fully frozen. lm_head at 8B is trained (or initialised differently) and not tied. Vocab is trimmed to drop redundant rows.

**Best explains:** obs 3 (cosines well below 1 but well above random), 4 (scales correlated, not equal — drift from initialisation), 5 (sign flips accumulating with depth, decreasing with size because larger models tolerate fewer flips per token of training), 7 (lm_head diverged because it received gradient signal), 8 (q/k_norm frozen; input/post_attn_norm trained), 11, 9, 10, 12.

**Doesn't comfortably explain:** 6 (why does 8B embedding *exactly* match the closed form to FP16). 2 (training under STE typically produces non-Bernoulli sign statistics; observed Binomial(127,0.5) is suspiciously clean).

**Discriminator vs H_b:** Check whether 1.7B/4B `embed_tokens` byte-equals the formula. If yes → tied-head means embed is post-training-modified there too, weakening "embed frozen" story. If no → embed was frozen at 8B but not at smaller sizes, which is odd.

## H_b — formula only for embed / q_norm / k_norm; everything else trained from a binary parameterisation

**Commits to:** Initialise the binary lattice however (random or `sign(w_base)`), keep a learned FP scale per 128-block, train end-to-end with binary forward + STE backward against a distillation target (Qwen3-base logits or text). Embed / q_norm / k_norm are *copied verbatim* from Qwen3-base via the closed form (embed) or directly (norms). lm_head is trained.

**Best explains:** 6, 7, 8 (the frozen-vs-trained split is exact), 2 (STE training from a pseudo-random init gives Bernoulli-like sign stats), 3, 5, 11, 12.

**Doesn't comfortably explain:** 4 (why would *trained* per-block scales correlate 0.62–0.73 with base `mean(|w|)` if scales are learned freely? Possible if scales were initialised from `mean(|w|)` and only lightly updated, but that's an extra commitment).

**Discriminator vs H_a:** Look at the joint distribution of (sign-disagreement-rate, scale-deviation) per block. H_a predicts they co-vary (blocks that drifted in sign also drifted in scale). H_b predicts scale drift is roughly independent of sign-flip rate.

## H_c — pure formula + tiny identity-head fine-tune

**Commits to:** Apply `sign(w) · mean(|w|_g)` to *all* matrix-heavy tensors of Qwen3-base. Then fine-tune only lm_head (8B) and the two trainable norm families on a small persona / distillation corpus. No gradient updates to matrix-heavy weights at all.

**Best explains:** 6, 8 (exact frozen/trained split), 2 (no STE artefacts), 4 (scales are exactly `mean(|w|)`), 9, 10, 12.

**Doesn't comfortably explain:** 3 (cosines as low as 0.34 — pure projection would give higher row cosines), 5 (sign disagreement rates of 22–32% on matrix-heavy weights — projection gives 0% disagreement by construction), 11 (size-dependent drift is unexplained if matrix weights are untouched).

**Status:** **Falsified by obs 5 for matrix-heavy weights.** Kept on the list because it cleanly explains the embed/norm subset and bounds the "minimum work" baseline a hypothesis must explain past.

## Reading rule

When citing one of these hypotheses, name the obs it explains and the obs it doesn't. Do not blur "the technique" with "H_a" — H_a is one candidate. The next experiments should target the discriminators listed above.
