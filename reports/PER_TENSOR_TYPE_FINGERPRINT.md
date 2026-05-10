# Per-projection-type fingerprint for a Bonsai-style reproduction (8B)

Derived from `reports/local-8B/20_*` through `27_*`. Each row is one
projection type; the columns are the byte-level signatures a
reproduction targeting Bonsai-8B should expect to land on.

## Master fingerprint table

```
type   sign-match  scale-CV   row-amp     median-     amp_ratio   depth pattern of:
       vs teacher  (mean)     vs-teacher  ratio       CV per-     per-block CV /
       (mean)                 (Pearson)   per tensor  tensor      sign-match
       across L              (8B)         (s_bonsai/  CV          across L
                                          mean(|w|))  (typical)
----   ----------  --------  ----------   ----------  ----------  -------------------
embed   0.9994     n/a       (n/a)        n/a         n/a         single tensor; byte-equal formula(teacher)
lm_head 0.8992     n/a       n/a          n/a         n/a         signs from teacher; scales recomputed
q       0.730      0.21      +0.80        ~1.4-1.9    0.30        CV ↑ depth (1.21x); sign-match ↓ slowly
k       0.733      0.20      +0.70        ~1.5-1.9    0.30        CV ↑ depth (1.23x)
v       0.792      0.13      +0.70        ~2.0-2.3    0.11        CV ↑ depth (1.51x); HIGHEST sign-match
o       0.764      0.20      +0.86        ~1.5-2.0    n/a         CV ↑ depth (1.38x); o is 'anomalous' on disturbance-axis
gate    0.729      0.30      +0.42        ~1.5-2.2    0.14        CV ↓ depth (0.54x; teacher-inherited)
up      0.735      0.25      +0.42        ~1.5-2.4    n/a         CV ↓ depth (0.39x; teacher-inherited)
down    0.777      0.13      +0.82        ~1.4-1.5    0.07        CV roughly flat across depth
```

(`n/a` = not measured systematically across depth in this batch; the
median values come from sampled tensors.)

## Reproduction targets per projection type

For each type, what a recipe-implementation should aim for:

### `q_proj` (and similarly `k_proj`)

- Sign agreement with teacher: target ~73% mean across all layers,
  range 70-77% per-layer.
- Per-block scale CV: ~0.2 (early layers around 0.18, late around 0.25 —
  rising with depth).
- Per-row mean s_g should correlate with per-row mean(|w_teacher|) at
  Pearson ~0.8 — initialise per-block scales near
  `mean(|w_teacher|_g)` and let the SGD perturb them.
- Per-tensor amplification factor: ~1.5-1.9× the RMSE-optimal scalar
  for the chosen sign pattern.

### `v_proj`

- Highest sign-match of the matrix-heavy projections (~79%).
- Tightest per-block CV (~0.13).
- Per-tensor amplification ~2.0-2.3× — strongest amplification of
  any matrix-heavy type.
- Per-row inherits ~70% from teacher.

### `o_proj`

- Sign-match ~76%.
- Per-row amplitude inheritance from teacher is high (Pearson +0.86)
  but the per-layer scale-CV does NOT track sign-match (Pearson +0.17;
  see `23_*`). This is the "anomalous" projection — its rows index
  hidden-dim, not head-dim like q/k/v.
- Per-block CV grows with depth (1.38× ratio).

### `mlp.gate_proj` and `mlp.up_proj`

- Lowest per-row Pearson with teacher (~0.42). The technique picks
  DIFFERENT rows to amplify than the teacher had loud, especially at
  late layers (L35 up: -0.22).
- Per-block CV is highest at early layers (gate: 0.44, up: 0.39) and
  falls with depth (gate ratio 0.54×, up ratio 0.39× early-vs-late).
  This depth shape is largely inherited from the teacher's natural
  weight distribution; the technique compresses the spread by ~2×.
- Sign-match has L1-3 dip (down to ~0.62) and recovers by L4.
- Per-tensor amplification ~1.5-2.4×.

### `mlp.down_proj`

- Sign-match ~78%.
- Per-row Pearson with teacher: +0.82.
- Tightest CV of any matrix-heavy projection (~0.07-0.14).
- Lowest amplification ratio (~1.4-1.5×).

### `embed_tokens`

- At 8B and 1.7B: byte-equal `formula(raw teacher)` — sign-match 99.9%,
  byte-match 99.9%.
- At 4B: 0.93 sign-match (heavier preprocess on this size's embedding).
- The deterministic formula path is force-by-data — no per-block SGD
  needed.

### `lm_head` (8B only — tied at 1.7B and 4B)

- Sign-match 89.9% (much higher than matrix-heavy 73-79%).
- Per-block scales recomputed (not formula-equal).
- Hassibi 2510.16250 prescribes that the last layer should NEVER be
  quantised; Bonsai violates this. Expect lm_head to need
  proportionally more SGD attention than other tensors per parameter.

## Three quick fingerprints a reproducer can verify

1. After your reproduction's per-block SGD step on q_proj at L0:
   Pearson(per-row mean(deployed s_g), per-row mean(|w_teacher|))
   should be ~0.80. If it's 0.95+ you haven't perturbed enough; if
   it's <0.5 you've perturbed too much.
2. For your reproduction's mlp.up_proj: Pearson should be ~0.42, NOT
   ~0.80. If your MLP looks like attention you haven't reproduced
   the disturbance.
3. Per-block amplification factor `s_bonsai / mean(σ · w_teacher)`
   should land at 1.5-2.4 with low intra-tensor variance (CV typically
   < 0.4). If the ratio is wildly variable (CV >> 1) the optimisation
   is fitting weights, not activations.

## Cross-size deltas (what changes 1.7B → 4B → 8B)

```
type   1.7B mean   4B mean   8B mean
q       0.705       0.692     0.730
k       0.700       0.715     0.733
v       0.761       0.778     0.792
o       0.741       0.743     0.764
gate    0.704       0.714     0.729
up      0.701       0.713     0.735
down    0.735       0.759     0.777
```

Cross-size signal: bigger Bonsai = closer to teacher (~+3pp 1.7B→8B).
Per-projection ordering is preserved at every size.
