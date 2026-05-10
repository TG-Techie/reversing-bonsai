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
26_per_row_amplitude_match.txt     per-row Bonsai vs teacher: attention/down inherit (Pearson 0.7-0.86); MLP gate/up deviate (0.42)
27_per_block_amp_ratio.txt         per-tensor amplification factor ~constant (CV 7-40%)
```

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
