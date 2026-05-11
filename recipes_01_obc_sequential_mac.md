# Recipe 01 — OBC-style sequential per-layer quantisation (Apple Silicon)

> Self-contained spec for a 1.125-bit quantisation procedure that
> fits on a single Apple Silicon laptop with ~26GB usable RAM. Hand
> this file to a recreation agent along with a base FP16 model and
> a calibration corpus; the agent produces a Q1_0_g128 GGUF
> "subject". The headline success criterion is **behavioural
> similarity** under the eval agent, not byte alignment.

---

## 0. What this recipe IS / IS NOT

**IS**: an attempt to reproduce a Bonsai-style 1.125-bit
compression using activation-aware per-layer per-block sign+scale
optimisation in the OBC family (Optimal Brain Compression,
arXiv:2208.11580 / arXiv:2210.17323). This is the natural
descendant of GPTQ adapted to a 1-bit codebook. No full-network
backward pass; weights are decided layer-by-layer using calibration
activations. Tractable on a laptop.

**IS NOT**: byte-identical reproduction of PrismML's actual recipe.
Several mechanism families produce similar byte signatures; this
recipe is one such candidate. The eval agent decides success on
behaviour.

---

## 1. Hardware requirements (to be provisioned before agent starts)

- **Apple Silicon Mac, M3 Max or M3 Ultra**, with ≥36GB unified
  memory (≥26GB available to the workload). M3 Pro 32GB also
  workable but tight for bases > ~4B.
- **Free flash storage**: ≥100GB writable to `$OUTPUT_DIR`.
- **Optional**: a second machine or cloud node for inference
  sanity checks if local Mac throughput is uncomfortable; recipe
  itself doesn't require it.
- **Power**: plugged in. Long-running unattended.

## 2. Software requirements (pre-installed by user)

- macOS 14.6+
- Python 3.11 with `venv` or `uv`
- One of the following ML backends:
  - **Preferred**: `mlx ≥ 0.18`, `mlx-lm ≥ 0.18` (Apple's native
    framework; fastest on Apple Silicon).
  - **Fallback**: `torch ≥ 2.5` with MPS backend. Slower but
    sufficient.
- `transformers ≥ 4.45`, `safetensors`, `datasets ≥ 3.0`,
  `gguf`, `numpy`, `tqdm`, `huggingface_hub`.
- `llama.cpp` built with Metal + Q1_0 support, available at
  `$LLAMA_CPP_ROOT`. Verify with `llama-quantize --help`. If the
  upstream `ggml-org/llama.cpp` lacks Q1_0, build from
  `PrismML-Eng/llama.cpp` fork.
- Xcode Command Line Tools (`xcode-select --install`).
- `huggingface-cli` with `HF_TOKEN` if base is gated.

## 3. Inputs (paths provided as environment variables)

```
$BASE_MODEL_DIR          # directory with FP16/BF16 safetensors
                         # + config.json + tokenizer.*
$CALIB_DATA_PATH         # JSONL file: one calibration sequence per
                         # line, "text" field. Aim ~128 × 2048-token
                         # segments from C4 / WikiText-2.
$OUTPUT_DIR              # empty writable directory
$REFERENCE_BONSAI_GGUF   # OPTIONAL: PrismML's released Bonsai GGUF
                         # for matching teacher; used for byte-
                         # fingerprint indicators only
```

## 4. Q1_0_g128 format spec (output target)

Identical to Recipe 00 §4. In summary:
- 1 sign bit per weight + 1 FP16 scale per 128-element block along
  input dim.
- Decode: `w_i = scale_g · (2·sign_bit_i − 1)`.
- Block layout: input dim is the fast/contiguous axis.
- Non-quantised: RMSNorm / q_norm / k_norm stay F32.
- Embed_tokens AND lm_head: quantised to Q1_0.

## 5. Pipeline

Six phases. Each is a natural checkpoint boundary; Phase B/C also
checkpoint per-layer so an interruption loses at most one layer of
work.

```
Phase A  Optional light LoRA preprocess              (1-3 h, optional)
Phase B  Calibration activation collection           (30-90 min)
Phase C  Per-layer sequential OBC-style quantisation (4-12 h)
Phase D  Embed + lm_head quantisation                (<10 min)
Phase E  GGUF packing                                (<15 min)
Phase F  Self-test                                   (<30 min)
```

### Phase A — Optional light LoRA preprocess

**Purpose**: small restorative fine-tune to nudge base into a
sign-pattern slightly easier to quantise. PTQ1.61's reported
pre-quant LoRA step in miniature.

**SKIP this phase if any of**:
- Available RAM < 26GB and base ≥ ~4B parameters.
- Wall-clock budget is < 8h total.

In those cases use the base unchanged as input to Phase B.

**If executing**:
- Attach LoRA at `r = 16`, `alpha = 16`, `dropout = 0`,
  `target_modules = "all-linear"`.
- Tokenise `$CALIB_DATA_PATH` to 2048-token packed sequences.
- Optimiser: AdamW, `lr = 2e-4`, cosine schedule, 5% warmup.
- Steps: 2000. Batch size 4 with gradient accumulation to
  effective 16.
- Loss: cross-entropy + 0.5 × KL distillation against the
  unmodified base on the same batch.
- Mixed precision: bf16 (MLX) or fp32 (MPS fallback).

**Checkpoint**: every 500 steps,
`$OUTPUT_DIR/checkpoints/phase_a/step_${STEP}.safetensors` with
adapter weights only. At end, merge LoRA into base and save
`$OUTPUT_DIR/intermediate/lora_merged.safetensors`.

**Resume**: if `intermediate/lora_merged.safetensors` exists, skip
Phase A. If `checkpoints/phase_a/step_*.safetensors` exist but
final merge hasn't happened, resume from the latest step.

### Phase B — Calibration activation collection

**Purpose**: capture the input distribution each Linear actually
sees, so Phase C can choose signs+scales that minimise
reconstruction error in the right inner-product space.

**Procedure**:
1. Load the model (LoRA-merged from Phase A if available, else
   base) into MLX or PyTorch-MPS.
2. Tokenise the first `N_CALIB = 128` sequences from
   `$CALIB_DATA_PATH`, packed to 2048 tokens each.
3. For each transformer layer `ℓ` (in forward order), register a
   hook on each matrix-heavy Linear that records its **input**
   tensor (shape `(B, T, d_in)`).
4. Run forward, batched as memory allows.
5. For each (layer, tensor), accumulate `H = sum(x^T @ x)` over
   all collected tokens (`d_in × d_in` matrix). Discard the raw
   activations once `H` is updated; keep only `H`.
6. Subsample tokens if memory pressure: cap at 16k tokens per
   tensor for `H`.

**Memory math** (8B-class teacher): each `H` is `d × d` FP32. For
attention `d=4096`: 64MB. For MLP `d=12288`: 576MB. With 36
layers × ~7 tensors per layer = 252 H matrices. Mostly small;
biggest are the MLP `H`s. Keep them on SSD, load on demand in
Phase C.

**Checkpoint**: per-layer.
`$OUTPUT_DIR/checkpoints/phase_b/layer_${L}_H.npz` containing a
dict `{tensor_name: H_matrix}` for that layer. Layer `${L}` is the
index 0-based.

**Resume**: skip any layer whose `layer_${L}_H.npz` already exists.

### Phase C — Per-layer sequential OBC-style quantisation

**Purpose**: jointly decide sign-bits and per-block scales by
minimising activation-reconstruction error layer by layer.

**Per-tensor procedure** (run for every matrix-heavy Linear, in
forward order across layers):

For each Linear with weight `W ∈ R^(d_out × d_in)` and Hessian
`H ∈ R^(d_in × d_in)` from Phase B:

```
# Reshape W into blocks along input dim
B = d_in // 128
W_blk = W.reshape(d_out, B, 128)              # (d_out, B, 128)

# For each (output row, block) pair, choose (sign_bits, scale) to
# minimise ||(W_blk[r, b, :] - sign * scale) @ X_block||^2 over
# this block's column subset of X.
# Equivalent to a sub-problem: minimise (Δ_row_b)^T H_bb (Δ_row_b)
# where H_bb is the 128x128 sub-block of H corresponding to this
# block's columns and Δ = W - sign*scale.
for r in range(d_out):
    for b in range(B):
        w = W_blk[r, b, :]                    # (128,)
        H_bb = H[b*128:(b+1)*128,
                 b*128:(b+1)*128]             # (128, 128)
        s_init = np.mean(np.abs(w))           # formula init
        sign_init = np.sign(w)
        # Refine: alternate between fixing sign and optimising scale,
        # and fixing scale and optimising sign (Hessian-aware).
        sign, scale = obc_block_refine(w, H_bb, sign_init, s_init,
                                       max_iter=10)
        sign_bits[r, b, :] = (sign > 0).astype(np.uint8)
        scale_g[r, b] = float(scale)
```

**`obc_block_refine` inner loop** (alternating minimisation):

```
def obc_block_refine(w, H, sign, scale, max_iter=10):
    # Closed-form scale given signs: s* = (sign^T H w) / (sign^T H sign)
    # Sign refinement: flip the sign whose marginal improvement is
    # largest, until no flip improves the objective.
    for _ in range(max_iter):
        # Scale step
        denom = sign @ H @ sign
        if denom > 0:
            scale = float((sign @ H @ w) / denom)
        # Sign step (greedy descent)
        changed = False
        for i in np.argsort(np.abs(w))[::-1]:  # large magnitudes first
            delta_old = w - sign * scale
            sign_flip = sign.copy(); sign_flip[i] = -sign_flip[i]
            delta_new = w - sign_flip * scale
            if delta_new @ H @ delta_new < delta_old @ H @ delta_old - 1e-12:
                sign = sign_flip
                changed = True
        if not changed:
            break
    return sign, scale
```

This is a 1-bit specialisation of the OBC inner loop. The
alternating sign↔scale refinement is cheap (~ms per block).

**Important detail — activation propagation**:
After all tensors in layer `ℓ` are quantised, the next layer's
input distribution shifts (because previous layer outputs are now
quantised). To match what Bonsai's bytes suggest (depth-graded
coupling), we should propagate this shift. Two options:

- (a) **Static**: collect all `H` matrices once at Phase B against
  the FULL FP teacher, then use them for all layers in Phase C.
  Cheaper but ignores compounding error.
- (b) **Dynamic** (recommended): after quantising layer `ℓ`, re-run
  a small forward pass (~16 sequences) through layers `0..ℓ` (with
  quantisation applied) to collect *updated* `H` matrices for
  layers `ℓ+1`. This compounds error layer-by-layer the way
  OBC-style sequential quantisation is supposed to.

Pick (b) if memory + time allow; pick (a) otherwise. Document
which was chosen in `recipe_decisions.md`.

**Checkpoint**: per-layer.
`$OUTPUT_DIR/checkpoints/phase_c/layer_${L}_quant.npz` containing
all of that layer's `(sign_bits, scale_g)` for all 7 tensors.
After each layer is written, also update
`$OUTPUT_DIR/intermediate/partial_quant_state.safetensors` with the
quantised weights merged into a working model (for option (b)'s
forward pass).

**Resume**: skip any layer whose `layer_${L}_quant.npz` already
exists. If (b) is the chosen mode, the partial-quant state file
must match the latest completed layer; otherwise re-derive from
`layer_*_quant.npz` files.

### Phase D — Embed + lm_head quantisation

**Embed**: use the deterministic formula from §4
(`sign(W) · mean(|W|_per_128_block)`). For most Qwen3 sizes this
matches Bonsai byte-equal so the recipe should not over-fit it.

**lm_head**: if untied from embed, quantise it with the same OBC-
style procedure as Phase C, using H collected over the last
transformer layer's output (the lm_head's input distribution). If
tied to embed, copy the embed quantisation.

**Checkpoint**: `$OUTPUT_DIR/checkpoints/phase_d/embed_lm_head.npz`.

### Phase E — GGUF packing

Identical to Recipe 00 §5 Phase D. Assemble all
`(sign_bits, scale_g)` from Phase C+D plus F32 norm parameters
into a single Q1_0 GGUF via `gguf-py`.

**Output**: `$OUTPUT_DIR/subject.gguf`. Mark
`$OUTPUT_DIR/checkpoints/phase_e/packed.done`.

### Phase F — Self-test

Identical to Recipe 00 §5 Phase E. Run `llama-cli`, `llama-perplexity`,
compute byte-fingerprint indicators (if `$REFERENCE_BONSAI_GGUF`),
write `self_test_report.md`.

## 6. Outputs the agent must produce

```
$OUTPUT_DIR/
├── subject.gguf
├── self_test_report.md
├── run_log.txt
├── recipe_decisions.md          # OBC option (a)/(b), any other
│                                # ambiguities resolved
├── intermediate/
│   ├── lora_merged.safetensors  # only if Phase A was run
│   └── partial_quant_state.*    # only if OBC option (b) was used
└── checkpoints/
    ├── phase_a/                 # only if Phase A was run
    │   └── step_*.safetensors
    ├── phase_b/
    │   └── layer_*_H.npz
    ├── phase_c/
    │   └── layer_*_quant.npz
    ├── phase_d/
    │   └── embed_lm_head.npz
    └── phase_e/
        └── packed.done
```

## 7. Success criteria

Same as Recipe 00 §7. **Behavioural similarity** (judged by the
eval agent) is the headline; the recreation agent only self-tests
and reports.

## 8. Failure / ambiguity handling

Same as Recipe 00 §8.

**Mac-specific failure modes** to watch for:
- MLX OOM with cryptic Metal errors: if seen, lower
  `OBC option (b)` to `(a)`, or reduce the calibration-token cap
  in Phase B to 8k per tensor.
- Thermal throttling: long runs on AC power should be fine; if
  the kernel time per phase doubles vs initial estimate, check
  `sudo powermetrics --samplers smc` for CPU/GPU temperature and
  pause if at thermal limit.
- `mlx-lm` / `transformers` model loading discrepancies: prefer
  loading via MLX-LM's `load()`; if that fails for the specific
  base, fall back to PyTorch-MPS via `transformers.AutoModel*`.

## 9. Notes for the recreation agent

- This is **one** candidate recipe. Subsequent invocations may use
  different recipes; for THIS invocation, execute exactly this
  recipe.
- Behavioural similarity (judged by the eval agent) is the
  ultimate success criterion. Byte-fingerprint indicators are
  informational only.
- Resume aggressively from checkpoints. Phase C is the long phase;
  losing one layer of work is acceptable, losing all 36 is not.
- Total wall-clock budget: typically 6-18 hours. If Phase C alone
  exceeds 24 hours on an M3 Max with reasonable base size, halt
  and report — likely Phase B over-collected activations.
- The agent runs unattended on a laptop. Avoid prompts that need
  human input. Log to `run_log.txt`; surface any blocking errors
  by stopping the process so the user sees them in the next check-in.
