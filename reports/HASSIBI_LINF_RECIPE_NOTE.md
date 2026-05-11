# Hassibi-group ℓ∞ result: predictions and constraints for Bonsai's recipe

> Annotated reading of *One-Bit Quantization and Sparsification for Multiclass Linear Classification via Regularized Regression* — Ghane, Akhtiamov, Hassibi (Caltech), Feb 2024 / rev. Oct 2024. arXiv 2402.10474. PDF in `reports/related_papers/`. This note extracts each specific testable prediction from the paper, says how we tested it against Bonsai's bytes, and flags which prediction is the actual recipe-relevant signal vs which is just format-compatible.

## 1. The paper's setting (what is and is not analogous to Bonsai)

The paper analyses **multi-class linear classification** on a **Gaussian Mixture Model** with proportion `c` of the labels corrupted, in the **over-parametrised regime** (`d/n` fixed as `n → ∞`, with `d` being parameter dimension). The estimator is regularised linear regression:

```
min_W  ‖XW − Y‖²_F  +  λ Σ_ℓ f(w_ℓ)
```

where `f(·)` is a convex regulariser and `λ` is the regularisation strength. They study three cases: `f = ‖·‖²_2` (ridge), `f = ‖·‖_1` (lasso), `f = ‖·‖_∞`.

**What carries over to a transformer:** the per-row weight matrix can in principle be read as one classifier per output channel; *each row* `w_ℓ` of a `(out, in)` weight matrix is a "classifier" in this analogy. For a 1-bit-friendly recipe, applying ℓ∞-regularisation per row is the natural direct extension.

**What does not carry over:** the paper is for a single linear layer, not a deep network. The "data" `X` for a hidden layer in a transformer is the activations from the previous layer, which are not iid Gaussian and are themselves a function of the upstream quantised weights. So any direct application to a transformer requires extending past the paper's analytical regime — that's where the unpublished "Caltech IP" sits.

## 2. The five predictions, mapped to Bonsai

### P1 — Half-and-half sign distribution at the boundary

**Paper:** Theorem 4.6 + Corollary 4.7. As `λ → ∞`, exactly `ζ` of the weight coordinates concentrate at `+δ/λ` and `ζ` concentrate at `-δ/λ`, where `ζ → 1/2`.

**Bonsai byte-level test:** per-128-block positive count.

**Result:** mean pos_count per block = 64.0 (binomial null), tightly distributed across all tensors, all sizes. **Holds — but consistent with a uniform-random sign distribution under any procedure**, so confirmation is weak.

### P2 — All weights on the boundary at large λ

**Paper:** Section 5.3 figure: "percentage of weights on the boundary approaches 1" as `λ` grows.

**Bonsai byte-level test:** per-block fraction of weights at exactly `±s_g`.

**Result:** binary_frac ≈ 0.9999 across 13.4M groups at 1.7B; same at 4B and 8B. **Holds — but trivially, since Q1_0 *enforces* it by format.** This is force-by-format, not differential evidence.

### P3 — Sign carries all the information at large λ

**Paper:** "since the classification error doesn't change under rescaling, one can replace weights with their signs in the large-λ regime without affecting performance." So sign is the only information that survives at the limit.

**Bonsai byte-level test:** is `sign(w)` what the format keeps, with magnitude collapsed to a per-block scalar?

**Result:** yes — Q1_0_g128 stores sign + per-128-block scalar. **Holds — by format design.** Again, format-friendly to ℓ∞ but not differential evidence.

### P4 — Boundary value `δ/λ` is data-dependent, NOT teacher-weight-dependent

**Paper:** the boundary `δ/λ` is determined by the data statistics (means `μ_ℓ`, variance `σ²`, label-corruption `c`, number of classes `k`, dimension `d`, sample size `n`) — see Theorem 4.6 closed-form. There is **no relationship between `δ/λ` and any pre-existing weight matrix** (the paper has no "teacher" — the classifier is trained from scratch given data).

**Bonsai byte-level test:** is `s_bonsai_per_block` equal to `mean(|w_base|_per_block)` (the natural teacher-derived choice)?

**Result:** **NO — `0/53` matrix-heavy tensors at 8B have any block where `s_bonsai` byte-equals `mean(|w_base|)`** (`reports/local-8B/12_*`). And `s_bonsai` is NOT the RMSE-optimal scale either (~2× larger; `reports/local-8B/13_*`). **This is the differential evidence.** The deployed scales come from *something other than weight statistics* — exactly what the paper's data-dependent `δ/λ` predicts.

### P5 — Boundary value depends on a noise / SNR scalar specific to that classifier

**Paper:** in the multi-class setting (Theorem 4.6), each class `ℓ` has its own `(μ_ℓ, σ², c)` and thus its own `δ_ℓ/λ`. So if you read each output row as a class, scales should vary per-row in a way driven by per-row activation statistics, NOT shared across rows.

**Bonsai byte-level test:** does Bonsai impose per-row uniformity of block scales (the strongest reading of "each row = one classifier")?

**Result:** **NO** — `reports/local-8B/18_*` shows that the per-row scale clustering ratio in Bonsai matches Qwen3-base's. Bonsai did not impose row-uniform scales.

**Reading:** the recipe likely doesn't apply ℓ∞ at the row level. It likely applies it at the per-128-block level — each block is its own "classifier" with its own `δ_block/λ`. That fits both the format (which has per-block scales) and the data (per-block scales vary).

## 3. The specific recipe constraint that follows

If a reproduction wants to mimic Bonsai's deployed scale assignment, the recipe-relevant claim from this paper plus our bytes is:

> Per 128-block, the deployed scale `s_g` is the ℓ∞-regime boundary `δ_g/λ` for an optimisation that takes that block's *training signal* (activations or distillation loss) — NOT the teacher's weight statistics — as its data input.

In the paper's multi-class linear classification setting, `δ_g` would be a function of:
- per-class mean separation `‖μ_ℓ - μ_ℓ'‖`
- noise variance `σ²`
- mislabelling rate `c`
- over-parametrisation ratio `d/n`

In a transformer setting, the analogous quantities would be the per-block activation statistics during training. Without rerunning the training, we cannot back out the specific values.

## 4. What this rules in and rules out

**Rules in:**
- A per-128-block scale-optimisation procedure where each block's scale is determined by *something other than its teacher block's weight statistics*. ℓ∞-regularised classification of the block's training signal is one such procedure.
- The `~2×` magnitude inflation observed (`reports/local-8B/13_*`): in the paper's regime, `δ/λ` is set by the data, not by ‖w_teacher‖, so there is no reason it should equal the RMSE-optimal scalar onto teacher weights. Inflation past RMSE-optimal is the *expected* behaviour for an information-theoretically-correct boundary.

**Rules out (or at least: does not explain):**
- Per-row scale uniformity (section 18 test).
- The depth-erratic predictability of `s_bonsai` from base statistics (section 16 — base statistics predict scales well at L0 q_proj r²=0.75 but fail at L6 mlp_gate r²=0.02). The paper is single-layer; depth-dependent behaviour comes from the deep-network extension that's the unpublished IP.

## 5. Connection to the other Hassibi paper (2510.16250 — "1-Bit RF")

That paper's claim — "1-bit quantising every layer except the last is asymptotically lossless in the random-features model" — predicts our **observation 7** (lm_head is treated specially at 8B: separate tensor, signs largely from base, scales recomputed). The two papers together suggest the unpublished recipe extends ℓ∞-1-bit from the linear-classifier regime to deep networks **with the last layer special-cased**.

## 6. What's still missing for a faithful reproduction

- The exact loss / training signal used at each layer.
- Whether the procedure is single-pass forward (GPTQ-family) or iterative (QAT-family) — the depth-erratic r² favours forward-sequential, but isn't decisive.
- The identity-shaping step (force-by-data via inference test).

## 7. Reading rule (carry-over)

When citing this paper as "the theoretical anchor for Caltech IP", the only differential evidence is **P4**: scales are data-dependent, not teacher-weight-dependent. P1, P2, P3 are format-side and would hold under any 1-bit format. P5 is rejected at the row level. The unpublished step is the extension from single-layer linear classification to deep networks; the bytes alone can't tell us what that extension is.
