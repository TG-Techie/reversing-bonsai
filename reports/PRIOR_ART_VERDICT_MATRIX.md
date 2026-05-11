# Prior-art verdict matrix — 9 techniques scored against Bonsai-8B's bytes

> Constructed from 9 worktree-isolated, fresh-context sub-agents, each given the same set of byte-level Bonsai facts and asked to read one PDF in `reports/related_papers/`. Each agent returned a verdict on whether the technique's algorithmic predictions match the bytes, where they match, and where they don't. This document consolidates the verdicts. Synthesis at the end is mine.

## Score card

The columns are the 6 strongest empirical findings from `ASSUMPTIONS_AUDIT.md §1` and the cross-size patterns:

- **F1:** embed = naive Q1_0(teacher) byte-equal at 8B (and 1.7B; force-by-data)
- **F2:** matrix-heavy signs match teacher 75–80%, NOT 100% (ruling out direct sign-quant)
- **F3:** matrix-heavy per-block scales NEVER equal `mean(|w_base|_g)`; deployed scale is ~2× the RMSE-optimal `mean(σ·w_base)` (ruling out closed-form / RMSE-fit recipes)
- **F4:** per-block predictability of `s_bonsai` from base block features is ERRATIC across depth — q/k/v/o stable 0.45–0.80 but MLP r² bounces 0.02–0.86 non-monotonically (ruling out smooth depth-decay recipes)
- **F5:** lm_head at 8B is Q1_0-quantised with 89.9% sign match + recomputed scales (Bonsai-8B has untied lm_head; Hassibi RF predicts lm_head should NEVER be quantised, so this constrains the IP to include extra last-layer machinery)
- **F6:** input_layernorm has structured depth profile (peak excess 53× ULP at L0, identical to base by L35); q_norm/k_norm round-trip-tight everywhere

`✓` = consistent with the technique. `✗` = inconsistent. `~` = silent / not addressed by the technique. `▲` = partially compatible only with extra commitment / removal of a sub-mechanism.

| Technique | F1 | F2 | F3 | F4 | F5 | F6 | Verdict |
|---|---|---|---|---|---|---|---|
| **PTQ1.61** (2502.13179) | ▲ | ✓ | ✓ | ✓ | ✓ | ✓ | **★ broadly consistent** (drop salient-channel tier) |
| **Progressive 1-bit / BinaryLLM** (2508.06974) | ✓ | ✓ | ▲ | ~ | ✓ | ▲ | partial (per-row not per-block) |
| **BiLLM** (2402.04291) | ✗ | ✓ | ✓ | ✓ | ~ | ~ | only OBC-half consistent (salience tier ruled out by format) |
| **Hassibi ℓ∞** (2402.10474) | ✓ | ~ | ✓ | ~ | ~ | ~ | theoretical anchor (predicts inflated boundary; no algorithm) |
| **GPTQ** (2210.17323) | ✓ | ✗ | ✗ | ✗ | ~ | ~ | ruled out (smooth scales, ~100% sign match, RMSE-leaning) |
| **OneBit** (2402.11295) | ✗ | ▲ | ▲ | ~ | ~ | ✓ | ruled out (rank-1 a·b factorisation incompatible with per-block) |
| **Output-alignment** (2512.21651) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ruled out as standalone (role-conditioned, no RMSNorm signal) |
| **Hassibi 1-bit RF** (2510.16250) | ✗ | ✗ | ✗ | ~ | ✗ | ~ | Bonsai VIOLATES the central prescription (lm_head NEVER quant) |
| **STBLLM** (2408.01803) | ✗ | ✗ | ✗ | ~ | ✗ | ~ | ruled out by format (sub-1-bit with N:M sparsity + masks) |
| **Radio rate-distortion** (2505.03031) | ✗ | ✗ | ▲ | ~ | ✗ | ~ | semantic neighbour only (variable bit-depth + companding LUT) |

## Per-technique detail

### PTQ1.61 — broadly consistent (5/6 dimensions)

The closest published match. Specifically:

- F1: a closed-form fallback for embeddings is consistent with Bonsai's embed=formula byte-equality (the paper itself doesn't address embeds, but treating them via the deterministic formula on the LoRA-restored teacher fits cleanly).
- F2: PTQ1.61's pre-quant **restorative LoRA fine-tune on RedPajama** shifts the teacher weights *before* sign+scale extraction, so 25% of signs naturally drift from the original teacher. Matches.
- F3: per-block α is **SGD-learned** against an output-cosine + MSE objective on calibration activations — not closed-form on weights. The deployed α deviates from `mean(|w|)` by amounts dependent on calibration activations. Matches both the 0/53 byte-equality AND the ~2× RMSE-optimal inflation (NLC objective pushes α larger than MSE-only).
- F4: SGD-learned α convergence quality varies block-by-block depending on calibration-activation geometry; no smooth depth-monotonicity is predicted. Matches the erratic depth pattern.
- F5: lm_head receives the same SGD treatment, with calibration activations specific to the head. No row-uniformity imposed.
- F6: norms are not the optimisation target in PTQ1.61, but RMSNorm scales would naturally drift at points where the calibration activations expose them most. Plausible.

**The one structural inconsistency:** PTQ1.61 has a 4-bit salient-column tier accounting for ~0.0002 bits/weight, gated by a 1D per-column salience mask. Q1_0_g128 has no per-column heterogeneity — every weight is exactly 1 bit + shared scale. **Either PrismML uses PTQ1.61 *without* the salient tier, or that's the differentiator.**

### Progressive 1-bit / BinaryLLM — partial match

Has Binary-aware Initialisation (per-input-channel rescaling before sign extraction), 20-phase progressive `tanh` schedule, dual scaling. Compatible with the 25% sign drift, the inflated scales, and the embed/norm-frozen pattern. **Crucial gap:** uses per-row/per-tensor scales, NOT per-128-block. So the deployment format would differ — unless the technique is composed *with* a 128-block downsampling step at deployment. Also offers no explanation for the cross-size pattern (bigger Bonsai = closer to base).

### BiLLM — only the OBC-half is compatible

BiLLM's full pipeline has 4 components:

1. Salient/non-salient column split (~1–5% by Hessian-weighted saliency)
2. Residual binarisation: salient columns stored as `α_o·B_o + α_r·B_r` (TWO sign tensors)
3. Optimal break-point splitting non-salient weights into sparse/concentrated regions with their own scales
4. GPTQ/OBC block-wise error compensation

**Components 1–3 are STRUCTURALLY incompatible with Q1_0_g128** — single sign plane, single scale per block, no per-column heterogeneity. **Only component 4 is consistent.** Specifically, the byte signature of running OBC's column-by-column compensation inside a 1-bit grid (without BiLLM's salience tail-handling) would produce: ~75% sign agreement, scales ≠ `mean(|w|)`, ~2× RMSE-optimal magnitude, erratic-with-depth predictability of `s_bonsai` from base block features. All four match.

### Hassibi ℓ∞ (2402.10474) — theoretical anchor only

Predicts the *fixed point* of constrained optimisation: weights at the ℓ∞ boundary `±δ/λ`, with the boundary value `δ/λ` determined by data statistics (means, σ², mislabelling rate, sample size) — *not* by any teacher. **F3 (data-dependent scale, not teacher-weight-dependent) is the differential evidence for this.** Other predictions (P1: half-and-half sign distribution, P2: weights all on boundary, P3: sign carries all info) are format-side and would hold under any 1-bit format. Per-row clustering (P5) is **rejected** by `reports/local-8B/18_*`.

The paper provides NO algorithm — only the asymptotic structure of the optimum. So it is *not* a recipe; it is a justification for why the recipe should produce 1-bit-like outputs.

### GPTQ — ruled out

Predicts smooth depth-uniform scale predictability (Bonsai bounces erratically), near-100% sign agreement (Bonsai 75%), RMSE-leaning min-max scales (Bonsai is 2× inflated), doesn't natively reach 1-bit. **A GPTQ derivative for Bonsai would need most of the load-bearing GPTQ machinery replaced.** Only the layer-by-layer + column-by-column processing pattern survives, and that's already inherited via BiLLM/PTQ1.61.

### OneBit — ruled out

OneBit factorises `W = Sign(W) · diag(a) · diag(b)` — one scalar per output row × one scalar per input column. The deployed magnitudes lie on a rank-1 outer product `a_i · b_j`. **Q1_0_g128 has per-128-block scales, NOT per-row × per-input-col**, and Bonsai's scales don't factor that way (per-row scale variance ratio matches base, not the constant ratio OneBit predicts). Also OneBit leaves embeddings in FP16; Bonsai quantises them.

### Output-alignment (2512.21651) — ruled out as standalone

Predicts a **role-conditioned** signature: only `o_proj`, `down_proj`, `lm_head` (the "last FC of each block") receive output-alignment treatment; others use weight-alignment. Predicts **flat-within-role depth patterns**. Bonsai shows the *opposite* — strong within-role across-depth variability (mlp_gate r² swings 0.02 → 0.46 → 0.08 at different depths). Also **doesn't touch RMSNorm**, so cannot explain F6 (input_ln L0 peak).

### Hassibi 1-bit RF (2510.16250) — Bonsai VIOLATES the prescription

The theorem assumes hidden weights are i.i.d. *random* (not learned from a teacher); Bonsai's signs track teacher 75% (not random). The paper's **explicit central prescription:** "the last layer should NEVER be quantised in practice" because it requires re-training in full precision. **Bonsai-8B's lm_head IS Q1_0-quantised.** So Bonsai is not a direct application of this theorem; the unpublished IP must include additional machinery to make the last layer survive quantisation.

### STBLLM — ruled out by format

Sub-1-bit output (0.53–0.85 bpw) via N:M sparsity + region tags + dual matrices for salient rows. Q1_0_g128 cannot represent any of this — there is no zero state, no per-element mask, no region tags, single scale per 128-block (not per-channel). Force-by-format incompatibility.

### Radio rate-distortion — semantic neighbour only

The Bonsai whitepaper's "intelligence density" framing is rate-distortion-flavoured, and Radio does rate-distortion-optimised LLM compression. But Radio's algorithm produces **variable per-group bit-depths**, a 256-entry companding LUT, and post-quant bias corrections — **none of which appear in the Bonsai bytes.** Treat as framing-level inspiration, not algorithmic candidate.

## Synthesis

The empirical fingerprint of Bonsai-8B's bytes is most consistent with a *novel combination* not published in any single paper above:

1. **Pre-quant restorative LoRA fine-tune of teacher** (PTQ1.61's signature; explains 25% sign drift and embed=formula(LoRA-shifted-teacher) byte-equality).
2. **Closed-form Q1_0 quantisation of LoRA-shifted teacher embedding** (the deterministic formula path; explains F1).
3. **SGD-learned per-128-block scales** with output-cosine + MSE objective on calibration activations (PTQ1.61's α-learning, with 4-bit salient-column tier dropped because Q1_0 is uniform; explains F3 and F4).
4. **Layer-by-layer OBC-style error compensation** (BiLLM/GPTQ inheritance; consistent with the depth-erratic predictability — accumulated activation distortion is the technique's signal).
5. **lm_head receives the same Q1_0 + SGD-α treatment** (Hassibi RF says shouldn't survive; the unpublished Caltech IP enables it, plausibly through the LoRA + SGD machinery being strong enough to compensate).
6. **input_layernorm trainable**, peak update at L0 — explains F6, not addressed by any single paper (this is custom).
7. **Identity-shaping** via the LoRA fine-tune corpus content (explains the inference self-ID test).

The **Hassibi ℓ∞ result (2402.10474)** is the theoretical anchor — it justifies why a constrained optimisation with this structure should produce useful 1-bit weights with magnitudes at the boundary (i.e. 2× RMSE-optimal direction). It is not a recipe ingredient on its own.

## Open questions left after this round

- **Is the LoRA-restorative step actually present?** Testable: if we apply naive Q1_0 to a *short fine-tune* of Qwen3 on a small calibration corpus and the result has sign-agreement ~75% with raw base + scales close to `mean(|w_LoRA|)`, that supports the LoRA-restore hypothesis.
- **Does the SGD-α step reach Bonsai's specific scales?** Testable in principle by reproducing the optimisation; computationally expensive.
- **Is the input_ln L0 peak from the technique or from gradient flow during the SGD pass?** Testable by ablating which params are trainable.
- **Are the 4B numbers consistent with this picture?** We'd need to re-fetch 4B base + run the full audit. Outstanding.

## Recipe-extraction maturity

After this round, we have:
- 5 published techniques **structurally ruled out** by format alone or by direct contradictions with the bytes.
- 2 published techniques **partially compatible** if specific sub-mechanisms are dropped (BiLLM's salience tier, BinaryLLM's per-row scales).
- 1 published technique **broadly compatible** (PTQ1.61 minus salient-channel tier).
- 1 paper that **provides theoretical motivation** but no algorithm (Hassibi ℓ∞).
- 1 paper that **shares framing** but not algorithm (Radio).

That's a much sharper hypothesis space than where this PR started. The next experiment to constrain further: actually run a small reproduction of the candidate recipe on a tiny base model and see if the byte signature comes out the same.
