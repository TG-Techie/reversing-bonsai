# 5-minute summary

> What this repo concluded about Bonsai, after a full reverse-engineering
> session through May 2026. For details, see `RECIPE_HINTS.md`,
> `reports/REPRODUCTION_SKELETON.md`, and the numbered reports under
> `reports/local-{1.7B,4B,8B}/`. For the dependence chain ("which
> conclusion follows from which measurement"), see
> `reports/local-8B/README.md`.

## What we set out to do

PrismML released "1-bit Bonsai" — Qwen3-{1.7B, 4B, 8B} child models
quantised to 1.125 bits/weight (`Q1_0_g128`: 1 sign bit per weight,
1 FP16 scale per 128-weight group). The whitepaper credits
"proprietary Caltech IP". We wanted to know what that IP must (or
must not) include, by reading the deployed bytes.

## What the bytes uniquely say (force-by-data)

These are reproducible from the artifacts. Every number in this
section is anchored to a specific report under `reports/local-*/`.

1. **Lossless dequant**: `dequant(Q1_0) ≡ unpacked` to 1 FP16 ULP.
   The unpacked file carries no information beyond the GGUF.
2. **Channel ordering preserved**: best-row-permutation cosine
   matches identity to ±1e-3 across every layer. No permutation
   was applied.
3. **Sign distribution within each 128-block is statistically
   random**: ~63.5 transitions/block, lag-1 autocorr ≈ 0. No
   sortedness, no clustering.
4. **Embedding (1.7B and 8B) is byte-equal to** `sign(W_teacher) ·
   mean(|W_teacher|_per_128_block)` — the deterministic naive Q1_0
   of the teacher. At 4B the embedding has 7% sign drift (heavier
   preprocess at this size).
5. **Matrix-heavy per-block scales are NEVER byte-equal to**
   `mean(|W_teacher|_per_block)` (0/253 tensors at 8B). Median ratio
   `s_bonsai / mean(|W_teacher|_g)` is 1.3-1.8×. Median ratio
   `s_bonsai / RMSE-optimal` is 2× (where RMSE-optimal is
   `mean(σ_bonsai · w_teacher)`). So even given Bonsai's chosen
   signs, the scales are amplified beyond the L2-best value onto
   the teacher.
6. **Sign agreement with teacher** is ~73-79% per matrix-heavy
   projection at 8B; cross-size: bigger Bonsai = closer to teacher
   (~+3pp 1.7B → 8B). Per-projection ordering preserved at every
   size: `v > down > o > {q,k,up,gate}` with the bottom four within
   1pp of each other.
7. **Sign flips are MAGNITUDE-GRADED**: smallest |w_teacher|
   decile flips at near-random rate (~0.47); largest decile flips
   at 0.3-2.5%. Smooth monotone gradient. **Size-invariant.**
8. **Top-1% scale blocks cluster by ROW, not column**: only
   1.5-22% of rows hold all top-1% blocks, but ~100% of columns do.
   Per-output-channel amplification, not per-input-position.
9. **Per-row amplification follows teacher** for q/k/v/o/down
   (Pearson 0.7-0.86). For MLP gate/up, depth-varying: ~0.6 early,
   ~0.2 late. Late MLP gate/up deviates most from teacher.
10. **Per-block amplification factor is approximately constant per
    tensor** (CV typically 7-40%) — the technique applies a
    near-uniform per-tensor `× ~2.0` amplification, not a wildly
    block-specific one.
11. **Per-tensor-type LoRA-equivalent strength is NOT uniform**:
    v and down receive ~70-90% of the perturbation strength applied
    to q/k/o/gate/up. Explains v's and down's higher sign-agreement.
12. **Within-row block lag-1 autocorrelation ≈ 0**: per-block
    scales within a row are statistically independent of column-
    neighbours. No within-row spatial smoothness.
13. **Scale-CV and sign-match are negatively correlated** within
    a layer for 6 of 7 projection types (Pearson -0.23 to -0.90).
    Same per-layer "disturbance" axis for both signals. `o_proj` is
    the lone exception.
14. **q_norm and k_norm are byte-identical to teacher** at every
    layer 0-35 (≤0.6× BF16 ULP at peak). Frozen.
15. **input_layernorm has structured depth profile**: peak excess
    53× BF16 ULP at L0, decaying to byte-identical at L35.
16. **`post_attention_layernorm` ≤ 4× BF16 ULP everywhere**.
    Barely touched.
17. **lm_head at 8B (separately stored, not tied)**: 89.9% sign-
    agreement with teacher; per-block scales recomputed.
    Hassibi-RF 2510.16250 explicitly says the last layer should
    NEVER be quantised; Bonsai violates this.
18. **`(W_bonsai - W_teacher)` is NOT dominantly low-rank at high
    ranks**: SVD shows rank-128 explains only 8-14% of squared-
    Frobenius-norm; rank-1024 explains 66%. Pure LoRA-only step 1
    at typical rank 16-128 is RULED OUT (would explain ~95% at
    rank-128). **But small-rank components are NOT ruled out**:
    rank-16 fraction at L1-3 MLP is 2-3× the L0 baseline (1.89-2.97%
    vs 1.03-1.26%), consistent with a small-rank (r=8-32) LoRA
    component active at L1-3 MLP combined with a per-element step.
    The "approximately full-rank" framing applies to total delta
    norm; small additive components can still affect flip patterns
    by construction.
19. **Behavioural observation**: running each Bonsai size with
    `temp=0` and an empty system prompt produces deterministic
    self-identification as "Bonsai by PrismML, created by Babak
    Hassibi at Caltech". The chat template injects no model name.
    Some pipeline step encoded this identity.
20. **Per-block flip counts are over-dispersed vs Binomial**, with
    the over-dispersion **depth- and projection-type-dependent**.
    `attn_q` over-dispersed at every depth (1.7-3.2); `attn_v` rises
    1.1 → 1.8 with depth (this rise is NOT confound-checked against
    Qwen3-8B v_proj teacher block-CV across depth; treat as
    consistent-with-depth-graded-coupling, not byte-attested);
    `attn_o` 1.6-2.3; `mlp.gate/up` are **U-shaped** (10-13 at
    L1-L3 → 1.1-1.4 mid → 2.2 at L35); `mlp.down` stays modest
    1.0-1.5. A Gaussian-noise control simulator gives 0.9-1.17
    across the magnitude deciles for q L0 — confirming the q L0
    over-dispersion is not just first-order Gaussian-noise artifact.
    Teacher-SIGN-blockstruct confound check shows teacher signs
    within blocks are i.i.d. (over-dispersion ~1.0 with shuffle
    controls matching to ±0.014). **But the teacher-MAGNITUDE
    confound at L1-3** (`45_*`) shows the L1-3 spike is partially
    inherited from Qwen3-8B's specific teacher block-heterogeneity
    at those depths. Reports `local-8B/37_*`, `38_*`, `40_*`, `45_*`.
21. **The depth-growing block coupling at MLP is NOT explained by
    growing LoRA rank**: full SVD across 5 depths × {gate, up} shows
    rank-128 % of squared-Frobenius-norm is essentially flat across
    L0/L9/L18/L27/L35 (7.5-9.05%, 1.6pp total spread). If a
    depth-graded LoRA rank produced the depth-growing coupling,
    rank-128 would rise substantially with depth. It doesn't. The
    over-dispersion's relative range (1.07-2.25 over the same grid)
    is over 100× the SVD's relative spread. Report `local-8B/39_*`.
22. **L1-L3 MLP "disturbance spike" is 8B-specific AND partially a
    teacher artifact** — at 8B the over-dispersion at L1-L3
    `mlp.gate`/`mlp.up` reaches **10-13**. SVD at L1-L3
    (`local-8B/41_*`) rules out heavy-rank-128 LoRA. Cross-size at
    1.7B (`local-1.7B/42_*`) shows the spike does NOT replicate:
    1.7B's L1 MLP over-dispersion is only 1.5-2.5. **A 4th
    confound check (`local-8B/45_*`) revealed that Qwen3-8B's
    teacher gate weights at L1-3 have ~10× higher cross-block CV
    and ~30× higher within-block CV variability than other depths.
    Random Gaussian noise applied to L1 teacher (calibrated to
    L1's flip rate) gives over-dispersion ~13 — close to Bonsai's
    actual 10.27.** So the "L1-3 spike" is substantially explained
    by Qwen3-8B's intrinsic teacher block-heterogeneity at those
    layers, NOT solely by Bonsai's recipe applying a special
    mechanism. A separate L1-3 rewrite step uniformly applied
    across sizes is still RULED OUT, but the previous reading
    "the recipe makes block-coherent decisions at L1-3" is
    substantially weakened. **Verifier-5 caveat**: an earlier
    framing in `45_*` that "Bonsai REDUCES the teacher's natural
    over-dispersion via per-block scale tuning" was over-attributed
    — the underlying Gaussian-noise simulation doesn't model
    per-block scaling, so the direction of recipe contribution at
    L1-3 (homogenising vs amplifying) is NOT byte-attested. What
    IS byte-attested: Qwen3-8B teacher structure is necessary and
    largely sufficient for the spike; the residual gap from
    simulation has multiple possible explanations.
23. **The L1-3 spike is MLP-specific** — attention at L1-3 shows no
    spike. Per (22), this is now interpretable as primarily a
    teacher-structure asymmetry: q_proj teacher weights are uniform
    across all depths (block_CV ~0.2 throughout), while gate_proj
    teacher at L1-3 has block_CV ~1.0 (10× higher). So the
    "MLP-asymmetric L1-3 effect" might reflect Qwen3-8B teacher
    weight structure rather than a recipe choice. The recipe-
    attributable signal is much weaker than the over-dispersion
    numbers suggested. See `local-8B/45_*`.

## What we INFER (not byte-attested, but consistent)

1. The technique includes a sign-quantisation step driven by
   output-loss (or distillation), since the magnitude-graded flip
   pattern is what any output-loss-driven sign assignment produces.
2. The technique includes per-block scale optimisation against
   activation behaviour, since the inflated-2× scales are not what
   weight-distance minimisation produces.
3. The technique applies different perturbation strengths per
   projection-type (lighter on v/down).
4. The technique amplifies specific output channels (rows) more
   than others; row identity at amplification time is partially
   inherited from the teacher's natural row-amplitude profile,
   especially at attention.

## Mechanisms RULED OUT

Each of these is inconsistent with at least one byte signature:

- **Pure-formula recipe** (apply `sign · mean(|w|_g)` to teacher):
  fails to produce the 25% sign drift; would give 100% sign-match
  with teacher.
- **Per-row uniform scales** (ℓ∞-min-norm interpretation): per-row
  scale variance ratio in Bonsai matches teacher's, not lower.
- **GPTQ vanilla**: predicts smooth depth-uniform scales (Bonsai
  bounces erratically), near-100% sign agreement (Bonsai ~75%),
  RMSE-leaning min-max scales (Bonsai 2× inflated).
- **OneBit factorisation**: predicts rank-1 `a_i · b_j` per-element
  scale structure. Bonsai's per-block scales don't factor that way.
- **Output-alignment alone**: predicts role-conditioned signature
  (only `o`/`down`/`lm_head` aligned). Bonsai shows within-role
  depth variability inconsistent with this.
- **STBLLM**: outputs sub-1-bit with N:M sparsity + region tags.
  Q1_0_g128 cannot represent any of this.
- **Pure-LoRA-only step 1** (at typical LoRA rank 16-128): SVD
  shows the delta is approximately full-rank; LoRA at rank-128
  would explain ~95% of squared-norm at rank-128; Bonsai's delta
  explains only 14%.
- **Pure i.i.d. element-wise noise on teacher**: matches the
  first-order magnitude-graded flip pattern but produces Binomial
  per-block flip counts (over-dispersion ~1.0). Bonsai's actual
  flip counts are over-dispersed at q every depth (1.95-2.67) and
  at deep MLP (>2.0 at L35). Some block-coherent component is
  required.
- **A uniform single-rank LoRA component** of any rank:
  rank-128 % of squared-Frobenius-norm of the delta is essentially
  flat (7.5-8.8%) across MLP depth, while block-coupling grows
  from ~1.1 (L0) to >2.0 (L35). The growing coupling cannot come
  from growing LoRA rank.

## Mechanisms still consistent with the bytes

- **PTQ1.61's full pipeline** minus its 4-bit salient-channel tier
  (which Q1_0_g128 cannot represent). 5/6 byte signatures match.
- **LoRA preprocess + full-rank SGD**: LoRA contributes a low-rank
  component, SGD adds the full-rank residual. Combined delta is
  approximately full-rank.
- **Pure QAT with STE**: per-element gradients with output loss,
  no rank constraint, produces near-full-rank delta and magnitude-
  graded flips.
- **OBC-style activation-aligned sign assignment**: signs chosen to
  minimise layer-output error given calibration activations.
- **A SwiGLU-sensitivity-weighted parallel pass** (suggested by
  verifier-3): a parallel-per-block recipe with a loss weighted by
  SwiGLU's input-gradient amplification would over-correct gate/up
  at early layers naturally, without requiring sequential ordering.
- **MLP-only small-rank LoRA (r=8-32) component plus a per-element
  step**: rank-16 fraction at L1-3 MLP is 2-3× the L0 baseline,
  consistent with a small LoRA component active there. Combined with
  full-rank SGD, the full delta would still look approximately
  full-rank.

The bytes don't discriminate among these alternatives. The strongest
*theoretical* anchor is the Hassibi-Akhtiamov-Ghane ℓ∞ result
(arXiv:2402.10474): in the over-parametrised regime with appropriate
regularisation, weights concentrate at two opposite-sign extremes
with magnitudes inflated past L2-optimal. That's the fixed-point
structure Bonsai's bytes match. The specific algorithm that produces
it is unspecified by the published theory.

## Open questions

- Whether the technique is QAT-style with STE, OBC-style sequential
  Hessian-aware quantisation, LoRA + SGD, or some hybrid.
- The exact loss function and its layer/head weighting.
- The training-data distribution.
- The schedule and ordering of quantisation vs identity-tuning.
- Why the 4B size-of-the-Bonsai-family received heavier embed
  preprocess than 1.7B and 8B.

## What this repo provides for someone trying to recreate Bonsai

1. A reproduction skeleton (`reports/REPRODUCTION_SKELETON.md`)
   with **12 falsifying tests** for the byte fingerprint (tests 8-12
   added in this batch from over-dispersion findings).
2. A per-tensor-type fingerprint (`reports/PER_TENSOR_TYPE_FINGERPRINT.md`)
   with target numbers for each projection type.
3. A nine-paper prior-art verdict matrix
   (`reports/PRIOR_ART_VERDICT_MATRIX.md`).
4. Streaming audit tools (`scripts/streaming_formula_audit.py`,
   `scripts/cross_tensor_overdisp.py`, `scripts/depth_overdisp_sweep.py`,
   `scripts/depth_svd_mlp.py`, `scripts/teacher_sign_blockstruct.py`,
   `scripts/attn_svd_l1l3.py`).
5. Quantitative reproduction targets:
   - `σ ≈ 1.25× teacher mean(|w|)` perturbation magnitude at 8B
   - 2× amplification factor
   - Magnitude-graded sign-flip curve (d1 ~0.47, d10 ~0.025)
   - Per-block over-dispersion 1.2-3.4× at q L0 8B
   - Full 36-layer over-dispersion depth profile (U-shaped at MLP)
   - L1-3 MLP-asymmetric, 8B-specific spike (must NOT replicate at
     1.7B uniformly)
   - Approximately full-rank delta everywhere (rank-128 < 28%)
6. **Eval-framework replication guide** (`reports/EVAL_REPLICABILITY.md`)
   — how to run PrismML's published-numbers harness on M3 Max via
   EvalScope + llama.cpp Metal + Gemini Flash Lite judge.

## How a reproduction would be tested

Given the verifier-3 corrections, the testable byte signature is:

- All 12 tests in `REPRODUCTION_SKELETON.md` pass.
- The reproduction's deployed Q1_0_g128 GGUF, run through the
  EvalScope + Gemini-judge harness, lands within ~2pp of Bonsai-8B's
  70.5 average (or matches the 8.8-pt gap to Qwen3-8B).
- The reproduction's byte signature is **MLP-asymmetric and
  depth-graded** matching Bonsai's, without producing the L1-3 spike
  uniformly across all sizes.

A reproduction that fails any byte-signature test has the wrong
mechanism even if the eval numbers match. A reproduction that
matches all byte-signature tests has *probably* recovered the
recipe family — but the bytes don't uniquely select among several
alternatives (see "Mechanisms still consistent with the bytes"
above).

A reproduction following PTQ1.61's recipe (minus the 4-bit
salient-channel tier) on a different base, with appropriate
calibration, should produce a model whose byte fingerprint
substantially overlaps Bonsai's.
