# local-8B reports — index

Numbered reports anchoring claims in `RECIPE_HINTS.md v4`,
`PRIOR_ART_VERDICT_MATRIX.md`, `REPRODUCTION_SKELETON.md`, and
`HASSIBI_LINF_RECIPE_NOTE.md`. Each report is the byte-level
measurement that supports a specific claim.

```
01_metadata.txt                    GGUF metadata + tensor inventory at 8B
02_q1_0_analysis.txt               H3 sign-pattern stats (random, no clustering)
03_h2_layer{0,6,12,17,18,24,30,35}.txt   H2 row-permutation cosines
05_dequant_vs_unpacked.txt         (NA at 8B; unpacked not on disk)
06_mag_layer{0,6,12,18,24,30,35}.txt     magnitudes follow-up sample
08_sign_layer{0,6,12,18,24,30,35}.txt    sign-disagreement per-tensor sample
09_ptq_baseline.txt                naive PTQ baseline anchor
10_per_head_signs.txt              per-head sign-flip rates (older sample)
11_discriminator_q0.txt            joint sign-flip × scale-drift on q L0
12_formula_match_per_tensor.txt    53 tensors at L0-7: sign/scale/byte vs formula
13_rmse_optimal_scale.txt          deployed s_g ≈ 2× RMSE-optimal
14_per_block_feature_correlations.txt    base-features predict s_bonsai
15_depth_check_correlations.txt    same at L17/L35 (depth check)
16_depth_sweep_7layers_7types.txt  7×7 depth sweep — base feature predictability
17_norm_depth_full.txt             36-layer norm equality (q/k frozen, input_ln peaks at L0)
18_linf_per_row_clustering.txt     ℓ∞ per-row uniformity prediction (rejected)
20_full_36layer_audit.txt          full 36-layer streaming audit (per-tensor sign/scale/byte vs formula)
21_scale_distribution_evolution.txt  per-layer scale-CV; attention rises with depth, MLP gate/up falls
22_per_head_scale_evolution.txt    head identity NOT preserved cross-layer
23_cv_signmatch_correlation.txt    per-layer CV ↔ sign-match: single disturbance axis (6/7 projections)
24_top1pct_block_clustering.txt    top-1% scale blocks cluster by ROW, not column
25_base_vs_bonsai_cv_evolution.txt teacher's MLP-CV-falls is INHERITED; technique compresses MLP CV ~2×
26_per_row_amplitude_match.txt     per-row Bonsai vs teacher: attention/down inherit (Pearson 0.7-0.86); MLP gate/up deviate; depth-conditioned (~0.6 early to ~0.2 late at gate/up)
27_per_block_amp_ratio.txt         per-tensor amplification factor ~constant (CV 7-40%)
28_within_row_block_autocorr.txt   within-row lag-1 autocorr ≈ 0 (per-block independence within row)
29_signflip_by_magnitude.txt       sign flips MAGNITUDE-GRADED (d1~0.47, d10~0.025); size-invariant. NOT discriminatory of LoRA vs other mechanisms.
33_per_kind_lora_strength.txt      implied per-tensor-type sigma; v/down receive lighter LoRA than q/k/o/gate/up
34_svd_low_rank_test.txt           SVD of (W_bonsai - W_teacher) is NOT low-rank; RULES OUT pure-LoRA-only step 1
35_delta_row_concentration.txt     per-row delta concentration modest (Pearson +0.87 with teacher row-norm)
36_mlp_sgd_concentration.txt       [RETRACTED] — was a sigma-misspecification artifact (verifier catch 2)
37_block_flip_overdispersion.txt   per-block flip count OVER-DISPERSED 1.2-3.4x at q L0; Gaussian control gives ~1.0; rules out i.i.d. element-wise noise
38_cross_tensor_overdisp.txt       cross-tensor (depth/type) over-dispersion grid; pattern depth/type-dependent
39_depth_svd_mlp.txt               depth-resolved SVD on MLP gate/up — rank-128 % flat across depth (RULES OUT depth-graded LoRA rank)
40_full_depth_overdisp.txt         full 36-layer x 5-projection over-dispersion sweep — U-shaped MLP profile (L1-3 spike 10-13)
41_l1l3_svd_mlp.txt                L1-3 MLP SVD — rules out heavy-rank-128 LoRA at the spike
43_attn_svd_l1l3.txt               attention SVD at L0-3, L18, L35 — L1-3 spike is MLP-specific in BOTH metrics
44_within_block_ordering_synthesis.txt  within-block attn-before-MLP ordering hypothesis [DOWNGRADED by verifier catch 3]
45_l1l3_teacher_confound.txt       L1-3 spike is PARTIALLY a teacher-structure confound (catch 4); recipe REDUCES teacher-natural over-dispersion
```

(`local-1.7B/42_*` is the 1.7B cross-size validation of the L1-3 spike.)

## Quick claim → evidence pointer

If you want to verify a specific claim from `RECIPE_HINTS.md v4`,
this is where to look:

| Claim | Evidence |
| --- | --- |
| embed = formula(teacher) byte-equal | `12_*` (8B), `19_*` (1.7B in local-1.7B) |
| Matrix-heavy scales NEVER == mean(\|w_teacher\|) | `12_*` (0/53 tensors) |
| Scales ~2× RMSE-optimal | `13_*`, `27_*` |
| Erratic depth predictability of s_bonsai | `14_*`, `15_*`, `16_*` |
| q_norm/k_norm frozen everywhere | `17_*` |
| input_ln peak excess at L0 (53× ULP) | `17_*` |
| ℓ∞ per-row uniformity NOT imposed | `18_*` |
| Per-projection sign-match ordering at full network | `20_*` |
| MLP gate/up CV falls is inherited from teacher | `25_*` |
| Top-1% blocks cluster by ROW not column | `24_*` |
| Per-row amplification follows teacher for attn/down | `26_*` |
| Single disturbance axis for 6/7 projections | `23_*` |
| Per-tensor amplification factor near-constant | `27_*` |
| Head identity NOT preserved cross-layer | `22_*` |
| Sign flips magnitude-graded (d1~0.47, d10~0.025) | `29_*` |
| Per-tensor-type LoRA strength (v/down lighter) | `33_*` |
| Bonsai delta is NOT low-rank (rules out pure-LoRA-only) | `34_*` |
| Per-row delta concentration modest, proportional to teacher row-norm | `35_*` |
| Per-block independence within rows | `28_*` |
| Per-block flip count is OVER-DISPERSED (rules out i.i.d. noise) | `37_*` |
| Cross-tensor over-dispersion depth/type-dependent | `38_*` |
| Depth-graded LoRA rank ruled out (rank-128 flat across depth) | `39_*` |
| U-shaped MLP over-dispersion profile (L1-3 spike, L33-35 rise) | `40_*` |
| Teacher signs i.i.d. within blocks (confound check) | `40_*` |
| L1-3 spike not heavy-rank-128 LoRA | `41_*` |
| L1-3 spike is 8B-specific (1.7B doesn't show it) | `42_*` (in local-1.7B) |
| L1-3 spike is MLP-specific (attention shows no spike) | `43_*` |
| Within-block attn-before-MLP ordering [WEAKENED by verifier-3] | `44_*` |
| L1-3 spike is partially a teacher-structure confound | `45_*` |
| Recipe REDUCES teacher-natural over-dispersion at L1-3 | `45_*` |
