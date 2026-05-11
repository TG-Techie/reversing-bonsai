# Recipe 00 — LoRA preprocess + QAT-with-STE (cloud GPU)

> Self-contained spec for a 1.125-bit quantisation procedure. Hand
> this file to a recreation agent along with a base FP16 model and a
> calibration corpus; the agent should produce a Q1_0_g128 GGUF
> "subject" model whose **behaviour** lands near the published
> Bonsai gap-vs-base. Byte-fingerprint indicators are documented
> below but **the headline target is benchmark behaviour, not byte
> alignment**.

---

## 0. What this recipe IS / IS NOT

**IS**: an attempt to reproduce PrismML's "Bonsai-style" 1-bit
compression on an arbitrary base via a documented combination of
two published techniques (LoRA preprocess à la PTQ1.61
[arXiv:2502.13179] + QAT with straight-through estimator à la
BitNet [arXiv:2310.11453]). This is one of several recipes that may
be tested; treat this invocation as standalone.

**IS NOT**: a guarantee of byte-equivalent reproduction. The bytes
of PrismML's actual Bonsai are NOT byte-attestable from this recipe;
several alternative mechanism families produce the same byte
fingerprint and PrismML has not published their exact algorithm.
The success criterion below is **behavioural**, not byte-level.

---

## 1. Hardware requirements (to be provisioned before agent starts)

- **1× NVIDIA GPU with ≥80GB VRAM** (A100 80GB or H100 80GB).
  - For a base model > ~10B parameters: 2× such GPUs with FSDP /
    DeepSpeed ZeRO-3 sharding.
- **System RAM**: ≥256GB.
- **SSD scratch**: ≥300GB writable to `$OUTPUT_DIR`.
- **Network**: outbound HTTPS to PyPI, HuggingFace, GitHub Releases,
  GitHub Container Registry.
- **Cost ballpark**: $30-120 single-GPU rental for the whole run on
  a base model in the 1.7B-8B range (~12-48 wall-clock hours).
- **Suggested providers (any equivalent works)**: RunPod, Lambda
  Labs, Vast.ai, Modal, CoreWeave.

## 2. Software requirements (pre-installed by user)

Versions are minimums; newer is fine.

- Python 3.11 with `venv` or `uv`
- PyTorch 2.5 with CUDA 12.x build (`torch.cuda.is_available() == True`)
- `transformers ≥ 4.45`, `accelerate ≥ 1.0`, `peft ≥ 0.13`,
  `bitsandbytes ≥ 0.44` (optional; for 8-bit Adam if memory-tight),
  `datasets ≥ 3.0`, `safetensors`, `gguf` (the Python package from
  `pip install gguf`), `numpy`, `tqdm`, `huggingface_hub`.
- `llama.cpp` built with Q1_0 kernel support, available at
  `$LLAMA_CPP_ROOT`. Verify by running
  `$LLAMA_CPP_ROOT/build/bin/llama-quantize --help` and confirming
  `Q1_0` is one of the listed types. If not present, build from a
  PrismML fork that exposes the Q1_0_g128 kernel
  (e.g. `PrismML-Eng/llama.cpp`).
- `huggingface-cli` configured with `HF_TOKEN` if the base model is
  gated (set in `~/.cache/huggingface/token`).

## 3. Inputs (paths provided as environment variables)

The user sets these before launching the agent. The agent must NOT
hardcode paths.

```
$BASE_MODEL_DIR          # directory containing FP16/BF16
                         # safetensors + config.json + tokenizer.*
$CALIB_DATA_PATH         # JSONL file: one calibration sequence per
                         # line, with field "text" (preferred
                         # ~256 × 2048-token segments of C4 or
                         # WikiText-2)
$OUTPUT_DIR              # empty writable directory
$REFERENCE_BONSAI_GGUF   # OPTIONAL: path to PrismML's released
                         # Bonsai GGUF for the same teacher
                         # architecture/size; used only for
                         # byte-fingerprint INDICATORS, not
                         # success criterion
```

## 4. Q1_0_g128 format spec (output target)

Output is a GGUF file conforming to the following per-tensor layout
for every matrix-heavy weight (attention q/k/v/o, MLP gate/up/down,
embed_tokens, lm_head if untied):

```c
#define QK1_0 128
struct block_q1_0 {
    ggml_half d;             // FP16 scale, group-wise
    uint8_t   qs[QK1_0 / 8]; // 16 bytes -> 128 sign bits,
                             // LSB-first within byte
};                           // 18 bytes total per 128-element block
```

Decoded weight: `w_i = scale_g · (2·sign_bit_i − 1)`. Effective
storage: 1.125 bits/weight (1 sign + 16 bits scale / 128 weights).

**Block layout**: 128-element blocks along the INPUT dimension of
each Linear weight (the "fast" dim under row-major reshape).

**Non-quantised parameters**:
- RMSNorm `weight` parameters: keep in F32.
- `q_norm` / `k_norm` (Qwen3 family): keep in F32. Initialise from
  base.
- Embed_tokens: quantised to Q1_0 per format above.
- lm_head: quantised to Q1_0 per format above (whether tied or not,
  output a Q1_0 lm_head in the GGUF).

The agent should rely on `gguf-py` to construct the file rather
than hand-rolling bytes.

## 5. Pipeline

The procedure has five phases. Each phase produces at least one
checkpoint; the agent MUST be able to resume from any checkpoint if
re-invoked with the same `$OUTPUT_DIR`.

```
Phase A  LoRA pre-quant restorative fine-tune          (2-4 h)
Phase B  Per-block sign + scale initialisation         (<5 min)
Phase C  QAT with straight-through estimator           (8-40 h)
Phase D  GGUF packing                                  (<15 min)
Phase E  Self-test                                     (<30 min)
```

### Phase A — LoRA pre-quant restorative fine-tune

**Purpose**: shift base weights into a sign-pattern more amenable
to subsequent 1-bit quant. Inspired by PTQ1.61's reported
pre-quantisation LoRA step.

**Setup**:
- Attach LoRA adapters to every Linear in `{q_proj, k_proj, v_proj,
  o_proj, gate_proj, up_proj, down_proj, lm_head}` across all
  transformer blocks.
- LoRA rank `r = 64`, `alpha = 32`, `dropout = 0`,
  `target_modules = "all-linear"`. Use `peft.LoraConfig`.
- Freeze base weights. Train only LoRA adapters.

**Optimiser**:
- AdamW, learning rate `2e-4`, `beta1 = 0.9`, `beta2 = 0.95`,
  `weight_decay = 0.0`.
- Cosine schedule with 5% warmup.
- Gradient clipping at 1.0.
- Mixed precision: bf16 (preferred) or fp16.

**Data**:
- Tokenise `$CALIB_DATA_PATH` with the base model's tokenizer.
- Pack to 2048-token sequences (concat + slice).
- Stop training at the earlier of:
  - 10,000 optimiser steps, OR
  - Validation cross-entropy plateaus for 1000 steps on a held-out
    10% split of `$CALIB_DATA_PATH`.

**Loss**:
- Cross-entropy on next-token prediction over the calibration text.
- (Optional) Add a teacher-distillation term: KL divergence between
  the LoRA-adapted model's logits and the base model's logits on
  the same batch. Weight 0.5. Helps prevent the LoRA from drifting
  too far semantically. Recommended.

**Checkpoint**: every 500 steps, save
`$OUTPUT_DIR/checkpoints/phase_a/step_${STEP}.safetensors` containing
only the LoRA adapter weights + optimiser state. At end, write
`$OUTPUT_DIR/checkpoints/phase_a/final.safetensors` and merge the
LoRA into the base, saving the merged FP16 to
`$OUTPUT_DIR/intermediate/lora_merged.safetensors`. The merged file
is the input to Phase B.

**Resume**: if `phase_a/final.safetensors` exists, skip Phase A
entirely. If only mid-training checkpoints exist, resume from the
latest step number.

### Phase B — Per-block sign + scale init from formula

**Purpose**: produce the initial Q1_0 quantisation that Phase C
will refine.

**Procedure** (single pass, no training):

```python
for tensor in all_matrix_heavy_weights:
    # tensor is (out_dim, in_dim), row-major
    W = lora_merged[tensor]                            # FP32
    blocks = W.reshape(out_dim, in_dim // 128, 128)
    sign_bits = (blocks >= 0).astype(np.uint8)         # 0 or 1
    scale_g   = np.mean(np.abs(blocks), axis=-1)       # (out_dim, n_blocks)
    save(sign_bits, scale_g)
```

For `embed_tokens` and `lm_head`, do the same.

**Checkpoint**: write
`$OUTPUT_DIR/checkpoints/phase_b/init_quant.npz` containing per-tensor
`sign_bits` (uint8, packed LSB-first per 128 weights) and `scale_g`
(FP16) arrays.

**Resume**: if `init_quant.npz` exists, skip Phase B.

### Phase C — QAT with straight-through estimator

**Purpose**: jointly refine sign-bit and scale choices against an
output-reconstruction loss, so the deployed Q1_0 model behaves like
the (LoRA-merged) FP teacher.

**BitLinear definition** (replaces every matrix-heavy Linear):

```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, init_weight):
        super().__init__()
        # FP32 shadow weight; this is what gets trained
        self.weight_shadow = nn.Parameter(init_weight.float())
    def forward(self, x):
        W = self.weight_shadow
        blocks = W.view(W.shape[0], -1, 128)
        sign = blocks.sign()
        sign = sign + (sign == 0).float()  # map 0 -> +1
        scale = blocks.abs().mean(dim=-1, keepdim=True)
        W_q = (sign * scale).view_as(W).to(x.dtype)
        # Straight-through estimator: gradients flow through W_q to W
        W_ste = W + (W_q - W).detach()
        return F.linear(x, W_ste, bias=None)
```

**Initialisation**: load each `BitLinear.weight_shadow` from the
LoRA-merged FP16 in `intermediate/lora_merged.safetensors`. RMSNorm
/ q_norm / k_norm: trainable FP32, initialised from base.

**Training**:
- Loss: cross-entropy on `$CALIB_DATA_PATH` + KL distillation
  against the unmodified base model's logits (same KL weight 0.5).
- Optimiser: AdamW, learning rate `1e-5`, `beta1 = 0.9`,
  `beta2 = 0.95`, `weight_decay = 0`.
- Cosine schedule with 3% warmup.
- Batch size 16 sequences × 2048 tokens. Gradient accumulation as
  needed for VRAM.
- Mixed precision: bf16.
- Steps: 30,000. Re-evaluate after 10k; if validation CE has
  plateaued, stop early.

**Why STE works**: forward uses the discretised quantiser so the
network sees the deployed behaviour; backward pretends the quantiser
is the identity so gradients can update the shadow weights. The
shadow weights drift to positions where the quantiser's output
minimises loss. This is standard BitNet machinery; see
arXiv:2310.11453 §3.

**Checkpoint**: every 1000 steps,
`$OUTPUT_DIR/checkpoints/phase_c/step_${STEP}.pt` containing all
`weight_shadow` parameters, RMSNorm scales, optimiser state, RNG
state. Use `torch.save` with `_use_new_zipfile_serialization=True`.

**Resume**: if `phase_c/final.pt` exists, skip Phase C. Otherwise
resume from `phase_c/step_${MAX}.pt`.

### Phase D — GGUF packing

**Procedure**:
1. Load final `phase_c/final.pt`.
2. For each matrix-heavy `weight_shadow`, derive `sign_bits` and
   `scale_g` per Phase B's formula.
3. Pack into Q1_0 blocks per §4.
4. Write `$OUTPUT_DIR/subject.gguf` via `gguf-py`, including:
   - All Q1_0 matrix-heavy tensors
   - F32 RMSNorm, q_norm, k_norm weights (use base names from the
     teacher's GGUF metadata convention; reuse `gguf-py`'s `Qwen3*`
     mappings or whatever family applies)
   - All metadata fields the base model's config requires
     (vocab_size, embedding_length, attention_head_count,
     attention_head_count_kv, feed_forward_length, etc.)
   - Tokenizer (copy from base)

**Checkpoint**: `subject.gguf` itself is the artifact. Write
`$OUTPUT_DIR/checkpoints/phase_d/packed.done` (empty file) to mark
phase completion.

### Phase E — Self-test

**Sanity checks** (all run on the deployed Q1_0 GGUF):

1. **Load test**: `$LLAMA_CPP_ROOT/build/bin/llama-cli -m subject.gguf
   -p "Hello, " -n 10`. Should produce coherent tokens.
2. **Perplexity sanity**: run `llama-perplexity` on a 1-2k token
   held-out C4 slice. Record value. Should be < 2× base model's
   perplexity on the same slice (rough guardrail).
3. **Byte-fingerprint indicators** (informational; if
   `$REFERENCE_BONSAI_GGUF` provided):
   - Sign-agreement vs base across all matrix-heavy weights.
     Target range: 70-80% (Bonsai range).
   - Per-block scale ratio `s_subject / mean(|w_base|_block)`.
     Target median: 1.3-1.8.
   - Magnitude-graded sign-flip rate by decile of `|w_base|`.
     Target: d1 ≈ 0.4-0.5, d10 ≈ 0.0-0.05 (smooth monotone).
   - These are INDICATORS only — fail/pass on benchmarks via the
     eval agent.

**Output**: `$OUTPUT_DIR/self_test_report.md` with the numbers above
and a one-paragraph summary.

## 6. Outputs the agent must produce

```
$OUTPUT_DIR/
├── subject.gguf                     # final deployable artifact
├── self_test_report.md              # sanity numbers
├── run_log.txt                      # phase-by-phase progress log
├── recipe_decisions.md              # any ambiguities resolved
│                                    # and the chosen defaults
├── intermediate/
│   └── lora_merged.safetensors      # mid-pipeline artifact
└── checkpoints/
    ├── phase_a/
    │   ├── step_*.safetensors
    │   └── final.safetensors
    ├── phase_b/
    │   └── init_quant.npz
    ├── phase_c/
    │   ├── step_*.pt
    │   └── final.pt
    └── phase_d/
        └── packed.done
```

## 7. Success criteria (HEADLINE: behavioural)

The eval agent (separate prompt, separate invocation) decides
success. This recipe's job is to produce `subject.gguf` and report
its self-test. The recreation agent should NOT make pass/fail
claims; that's the eval agent's role.

**Useful targets the agent can self-assess**:

- The model loads in `llama.cpp` and produces non-degenerate text.
- Perplexity on a held-out C4 slice is finite and < 3× base's
  perplexity.
- Self-test report shows byte-fingerprint indicators within Bonsai
  ranges (sign-agreement 70-80%, etc).

**Strong negative signals** that should be reported but NOT halt
the run:
- Sign-agreement vs base < 50% or > 95%: likely too much (or too
  little) refinement; recipe parameters may need adjustment.
- Per-block scale ratio < 0.8 or > 3.5: scale optimisation
  failed.
- Perplexity diverges (> 10× base): the model is broken; flag in
  report but DO produce the artifact for inspection.

## 8. Failure / ambiguity handling

If the agent encounters an ambiguity not resolved by this recipe:
1. Log the ambiguity to `recipe_decisions.md` with context.
2. Make a documented default choice based on the principle of
   minimal complexity (e.g., if asked "which dataset split?", use
   the first one available).
3. Continue.

If the agent encounters a hard failure (OOM, CUDA error,
infinite loss, NaN), DO NOT silently retry with degraded settings.
Stop the run, write a failure report to `run_log.txt`, and surface
the error.

## 9. Notes for the recreation agent

- This is **one** candidate recipe. You may be asked to apply
  different recipes in subsequent invocations — your job for THIS
  invocation is to execute exactly this recipe.
- Behavioural similarity to a reference Bonsai is the ultimate
  success criterion (judged by a separate eval agent). Byte
  signatures here are informational only — do not optimise toward
  byte-match if it costs perplexity.
- Resume from checkpoint if `$OUTPUT_DIR/checkpoints/` already has
  contents. Re-running this recipe with the same `$OUTPUT_DIR`
  should be idempotent past completed phases.
- Total wall-clock budget: typically 12-48 hours. If you see
  Phase C alone exceeding 60 hours, something is wrong; halt and
  report.
