# Glossary

Terms and acronyms used in `FINDINGS.md`, `reports/bonsai-1.7B/MINI_REPORT.md`,
and the script docstrings.

## Hypothesis labels (this repo's internal numbering)

- **H1** — *Lossless dequant.* The unpacked FP16 file equals
  `dequantize_row_q1_0` of the Q1\_0 GGUF, element-wise.
- **H2** — *No channel permutation.* Bonsai's row/column ordering matches
  Qwen3-base; no graph-equivalent permutation was applied between the base
  and the quantized model.
- **H3** — *Sign sortedness inside a block.* Within each 128-element Q1\_0
  block, the sign pattern is structured (sorted, run-length-friendly, or
  otherwise non-random).
- **H4** — *Input-column permutation* (added later). Even if rows match,
  input columns may have been reordered before quantization to cluster
  like-magnitudes into the same 128-block — which is graph-equivalent if
  applied consistently to the residual stream's producers and consumers.

These labels are local to this repo and don't appear in PrismML's
whitepapers. They're the questions we set out to answer, not their claims.

## Things being compared

- **Bonsai-Q1\_0** — `prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf`. The
  deployed 1-bit weight file; ~237 MB for the 1.7B model. Storage: one
  sign bit per weight + one FP16 scale per 128-weight group.
- **Bonsai-unpacked** — `prism-ml/Bonsai-1.7B-unpacked/model.safetensors`.
  An FP16 dump. Empirically (this repo, H1) it is exactly
  `dequantize_row_q1_0(Bonsai-Q1_0)` cast to FP16. Same information, three
  orders of magnitude more bytes.
- **Qwen3 base** — `Qwen/Qwen3-1.7B/*.safetensors`. The unmodified
  upstream Qwen3-1.7B model, BF16, that PrismML used as the starting
  point for Bonsai-1.7B.

## Format pieces

- **Q1\_0\_g128** — the GGUF block-quantization scheme PrismML uses. Block
  size 128, 1 sign bit per weight, 1 FP16 group scale. Sign bit `b ∈ {0, 1}`
  decodes to `w = s_g · (2b − 1) ∈ {±s_g}`.
- **scale `s_g`** (or `d`) — the FP16 number stored once per 128-weight
  group. In `ggml-quants.c` the reference encoder sets it to `mean(|x|)`
  over the group; the decoder reconstructs `±s_g` from the sign bits.
- **block / group** — interchangeable in Bonsai's context: a contiguous
  run of 128 weights along the fastest dim of a tensor, sharing one scale.
- **GGUF** — *GPT-Generated Unified Format*. llama.cpp's container format
  for quantized model files.
- **safetensors** — HuggingFace's plain-tensor container format. PRetty
  much "FP16 / BF16 numpy arrays plus a JSON manifest."
- **FP16** — IEEE 754 binary16, 1 sign / 5 exp / 10 mantissa bits.
- **BF16** — bfloat16, 1 sign / 8 exp / 7 mantissa bits. Same exponent
  range as FP32. Qwen3 ships in BF16.
- **ULP** — unit in the last place. The smallest representable spacing
  between adjacent floats at a given magnitude. "≤ 1 FP16 ULP" is the
  noise floor of an FP16 round-trip; agreement at that level is
  numerically identical.

## Architecture pieces (Qwen3, inherited by Bonsai)

- **Qwen3-{1.7B, 4B, 8B}** — Alibaba's dense decoder-only causal LMs.
  Bonsai's three sizes are direct children of these.
- **GQA** — *Grouped-Query Attention*. Multiple query heads share each
  key/value head. Qwen3-1.7B has 16 query heads × 8 KV heads, head\_dim 128.
- **RoPE** — *Rotary Position Embeddings*. Position info applied as 2D
  rotations of `(q, k)` pairs inside each head; this constrains channel
  ordering to respect head boundaries and pair structure.
- **RMSNorm** — *Root-Mean-Square layer norm.* Cheaper LayerNorm variant.
- **q\_norm / k\_norm** — Qwen3-specific per-head RMSNorms applied to `q`
  and `k` before the attention dot-product.
- **MLP / FFN** — *Multi-Layer Perceptron* / *Feed-Forward Network*. The
  per-token block in each transformer layer.
- **SwiGLU** — Swish-gated MLP: `down(silu(gate(x)) ⊙ up(x))`. Three
  matrices (`gate_proj`, `up_proj`, `down_proj`) instead of two.
- **FFN intermediate dim** — the "wide" inner width of an MLP. For
  Qwen3-1.7B, 6144. The dim that's freely permutable without changing
  model behavior, provided you permute `gate.rows`, `up.rows`, and
  `down.cols` together.
- **LM head** — final linear projection from the residual stream to vocab
  logits. In tied-embedding models, weight-shared with the input
  embedding matrix.
- **head\_dim** — the per-head feature width; 128 for Qwen3.

## Training / quantization techniques referenced (NOT empirically attested by us)

- **PTQ — Post-Training Quantization.** Take a finished FP16 model, snap
  each weight to the nearest grid point. Cheap, but typically loses
  significant accuracy below 4 bits/weight.
- **QAT — Quantization-Aware Training.** Train the model with the
  quantization in the loop: every forward uses the quantized weights, and
  gradients update an underlying FP shadow parameter via a
  *straight-through estimator* (treat the discontinuous quantizer as the
  identity in the backward pass). The model learns to live on the grid
  rather than being squashed onto it. BitNet-b1.58 is the canonical
  recent example for transformers.
- **Distillation / KD.** A smaller / lower-precision "student" model is
  trained to match the soft outputs (or hidden states) of a larger /
  higher-precision "teacher" — here, plausibly Qwen3 → Bonsai.

We *infer* that Bonsai used QAT (or a close cousin) because the empirical
sign-disagreement vs Qwen3 is too large for pure PTQ but the architecture
is byte-identical. We did not observe the training process and cannot
prove which specific recipe was used.

## Statistical terms

- **row vs column** in a 2D weight tensor `W` of shape `(out, in)`. Row =
  one output channel (length `in`). Column = one input channel (length
  `out`). The matmul `y = W x` dots each *row* with `x`.
- **Pearson correlation** — linear-correlation coefficient on raw values.
  Sensitive to scale and shift.
- **Spearman correlation** — Pearson on the *ranks* of the values.
  Sensitive only to monotone relationships; invariant to monotone
  reparameterization.
- **K-S statistic** — *Kolmogorov-Smirnov* two-sample test statistic.
  Largest gap between two empirical CDFs; 0 means identical
  distributions, 1 means disjoint supports.

## Things specific to this analysis

- **dequant ≡ unpacked** — shorthand for "the FP16 unpacked file is
  literally `dequantize_row_q1_0(Q1_0_GGUF).astype(np.float16)`."
- **identity row cosine** — row-by-row cosine similarity of two weight
  matrices in their as-shipped order, no permutation search.
- **best-perm row cosine** — row-by-row cosine similarity after a greedy
  nearest-neighbour search for an output-row permutation π that maximizes
  it.
- **per-block scale `s_g`** — the FP16 number stored per Q1\_0 block.
- **base group mean(|w|)** — for a given 128-element block, the average
  absolute value of the *Qwen3-base* weights in the same positions. The
  natural reference quantity to compare `s_g` against; in particular the
  reference Q1\_0 quantizer would set `s_g = mean(|x|)` exactly if Bonsai
  were a no-retrain sign-quant of Qwen3.
- **graph-equivalent permutation** — a coordinated reordering of channels
  that leaves the function the model computes unchanged. For
  transformers: reordering FFN intermediate dim across (gate.rows,
  up.rows, down.cols); reordering attention heads coherently; reordering
  the residual stream across embedding output, every Wq/Wk/Wv/Wo input
  column, every layer-norm, every MLP input column, and the LM head
  input.

## Terms added during the recipe-extraction work (May 2026)

- **formula** / **naive Q1\_0** — the deterministic Q1\_0 quantisation of
  a weight matrix: `w' = sign(w_base) · mean(|w_base|_per_128_block)`.
  When we say "Bonsai vs formula" or "Bonsai vs naive-quant" we mean the
  same thing.
- **RMSE-optimal scale** — for a fixed block sign pattern σ ∈ {±1}^128,
  the scalar `s* = mean(σ · w_base)` that minimises `‖s · σ - w_base‖²`.
  For matrix-heavy Bonsai blocks, `s_bonsai ≈ 2 · s*` — i.e. ~2× larger
  than what would minimise distance to base. Force-by-data; rules out
  any recipe that fits scales by L2 distance to teacher.
- **NLC** — *Negative Log Cosine*. The angular-alignment loss term
  `-log(cos(ŷ, y))` PTQ1.61 adds alongside MSE. Predicts inflated scales
  past the RMSE-optimum; consistent with our 2× ratio.
- **digester** — a worktree-isolated, fresh-context sub-agent given one
  prior-art PDF and the §1 byte facts, returning a verdict on whether
  the paper's algorithm matches those facts. See
  `reports/PRIOR_ART_VERDICT_MATRIX.md`.
- **streaming audit** — the OOM-safe pattern for full-network
  per-tensor analysis at 8B+: walk each base safetensors shard once;
  for each tensor in the shard, dequantise the corresponding Bonsai
  Q1\_0 from the GGUF on demand; compute metrics; free immediately.
  Peak memory is one tensor pair (~2GB at 8B) instead of all-tensors-
  at-once (which OOM'd this VM at 17GB).
- **autonomous-grant mode** — a multi-hour time window the user grants
  for self-directed work. The discipline is: pick a direction and
  execute, only stop when the budget runs out, periodic concrete-
  finding check-ins, and never ask "which direction next?" at batch
  boundaries. See CLAUDE.md.

### Algorithms / techniques referenced from prior art (NOT empirically attested in Bonsai's pipeline)

- **LoRA** — *Low-Rank Adaptation*. Add-and-train two low-rank matrices
  alongside frozen teacher weights. Used in PTQ1.61 as a pre-quant
  restorative step on a small calibration corpus.
- **STE** — *Straight-Through Estimator*. Replace a discontinuous
  (e.g. sign) function with its identity in the backward pass so
  gradients can flow through during training. Standard QAT machinery.
- **ℓ∞ / ℓ\_∞** — the max-norm regulariser. Hassibi-Akhtiamov-Ghane
  (arXiv:2402.10474) prove that under ℓ∞-min-norm regression with large
  λ, weights concentrate at two opposite-sign extreme values `±δ/λ`,
  giving a 1-bit-friendly fixed point with magnitude inflation past
  RMSE-optimal. The paper's setting is single-layer linear classifiers
  on Gaussian-mixture data; extending to deep transformers is the
  unpublished step that's likely the "Caltech IP".
- **OBC / GPTQ / BiLLM / OneBit / PTQ1.61 / BinaryLLM / STBLLM /
  Output-alignment / Radio** — published 1-bit / sub-2-bit quantisation
  techniques. Each has been digested by a sub-agent and scored against
  Bonsai's bytes. See `reports/PRIOR_ART_VERDICT_MATRIX.md` for verdicts;
  `reports/related_papers/` for PDFs.
- **calibration corpus** — a small sample of input text used by PTQ
  techniques to compute layer-wise activation statistics. Choice of
  corpus matters less than its diversity.

### Over-dispersion / block-coupling terms (introduced in 37_*-45_*)

- **Per-block flip-count** — for a given 128-block, the count of
  positions where `sign(W_bonsai) != sign(W_teacher)`. Ranges 0..128.
- **Binomial null / Binomial baseline** — the variance of per-block
  flip-counts under per-element-independent flipping at marginal
  rate `p`. Equals `128 · p · (1-p)`.
- **Over-dispersion ratio** — observed variance of per-block
  flip-counts divided by the Binomial baseline. = 1.0 means
  per-element-iid flips; > 1.0 means flips are correlated within
  blocks; < 1.0 means anti-correlated.
- **Block-coupled flips / block-coherent decisions** — a pattern
  where many positions within the same 128-block flip together
  (high over-dispersion). Can arise from low-rank LoRA structure
  or from per-block joint optimisation (OBC-style).
- **Teacher block-magnitude heterogeneity** — the cross-block CV
  of `mean(|w_teacher|_per_block)`. Qwen3-8B's `mlp.gate_proj` at
  L1-3 has 10× higher heterogeneity than at other depths — a
  property of the BASE model, not of Bonsai's recipe. Under any
  perturbation calibrated to the right flip rate, this heterogeneity
  alone produces high over-dispersion. Identified as the 4th
  confound in `local-8B/45_*`.
- **L1-3 disturbance dip / L1-3 spike** — at 8B, L1-L3 MLP gate/up
  show sign-match drop (to 0.62-0.65) and over-dispersion spike
  (10-13). NOT replicated at 1.7B (over-dispersion only 1.5-2.5).
  Partially explained by Qwen3-8B's teacher block-heterogeneity at
  those layers; Bonsai's recipe actually REDUCES the over-dispersion
  the teacher alone would produce.

### Process / discipline terms

- **verifier sub-agent** — a worktree-isolated, fresh-context
  general-purpose sub-agent given a recipe claim + data and asked
  to independently challenge it with explicit bias-naming. Caught
  4 confirmation-biased over-reaches in this session. See CLAUDE.md
  for the protocol and the four catches.
- **force-by-format / force-by-data / suggestion** — the three
  buckets for any empirical finding in `RECIPE_HINTS.md`. Format-
  level constraints (couldn't be otherwise) vs measurement-anchored
  constraints (reproducible from artifacts) vs patterns consistent
  with one specific reading.
- **confound check** — a control measurement establishing what
  "doing nothing" or "random noise" would produce, so the recipe-
  attributable signal can be separated from artifacts. The teacher-
  sign-blockstruct check (`40_*`) and teacher-block-magnitude
  heterogeneity check (`45_*`) are examples.
