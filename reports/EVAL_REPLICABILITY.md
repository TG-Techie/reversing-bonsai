# Replicating PrismML's Bonsai eval framework

PrismML's published eval setup (Appendix B of the 1-bit whitepaper) is
mostly reachable from open tooling. This note records what we'd need
to drop in to reproduce their numbers ± expected drift, with specific
attention to running it on local Apple-silicon hardware (M4 Max class)
rather than the CUDA H100 path they used.

## Their setup, summarised

```
component             version / config
--------------------- -------------------------------------------
harness               EvalScope v1.4.2 (Alibaba ModelScope, OSS)
inference engine      vLLM 0.15.1 (NVIDIA H100, FlashAttn-2)
attention backend     FLASH_ATTN (deterministic on H100 cap 9.0+)
batch invariance      VLLM_BATCH_INVARIANT=1
seed                  42
generation            greedy, temp=0.0, top_p=1.0
                      thinking-mode disabled (enable_thinking=false)
                      GPQA exception: temp=0.6, top_p=0.95, 10 samples
benchmarks            MMLU-Redux, GPQA-Diamond, MuSR, GSM8K, MATH-500,
                      HumanEval+, MBPP+, IFEval, IFBench, BFCLv3
extraction-fallback   Gemini 2.5 Flash Lite (temp=0.0) when rule fails
code sandbox          Docker python:3.11-slim
software              EvalScope hotfixes: MBPP+ test-suite + code
                      extractor patches (idempotent, in their repo)
```

For each benchmark:

```
benchmark      max-toks   judging
-------------- ---------  -------------------------------
MMLU-Redux       2048     rule + Gemini fallback
MuSR             2048     rule + Gemini fallback
GSM8K            2048     exact-match rule
IFEval           4096     OLLM strict (rule only)
IFBench          4096     OLLM strict (rule only)
BFCLv3 (13/17)   4096     AST + execution (rule only)
MBPP+            4096     pass@1 (Docker sandbox)
HumanEval+       8192     pass@1 (Docker sandbox)
MATH-500         8192     rule + Gemini fallback
GPQA-Diamond     8192     mean of 10 samples (sampling)
```

## What's drop-in for a local reproduction harness

- **EvalScope itself** is `pip install evalscope`. Their pinned
  `1.4.2` is on PyPI.
- **Benchmark datasets** are public (HF or hosted by EvalScope).
- **Code-execution sandbox** (`python:3.11-slim`) runs on macOS via
  Docker Desktop or Colima; benchmarks isolated per-test.
- **Rule-based scorers** (IFEval, IFBench, BFCL, GSM8K, code) are
  pure Python — no API calls.

## Two real gotchas for M4 Max replication

### 1. The inference engine

vLLM's CUDA backend is what their published numbers were run on.
There is no in-tree vLLM Metal backend. Two viable swaps:

- **llama.cpp Metal**: native Apple Silicon, fits Bonsai-Q1_0 GGUF
  and Qwen3-FP16 GGUF directly. PrismML's own llama.cpp fork
  (`PrismML-Eng/llama.cpp`) is the reference for the Q1_0_g128
  kernel.
- **MLX (Python)**: also native Apple Silicon. PrismML maintains a
  fork (`PrismML-Eng/mlx`) with their 1-bit kernel; until that's
  upstream, you'd use their fork or the 2-bit fallback path they
  use for the ternary release.

EvalScope speaks an OpenAI-compatible HTTP API. Both `llama-server`
and `mlx-lm`'s server expose that. So the harness layer is unchanged;
only the engine endpoint moves.

Expected drift: numerics differ between vLLM-FA2 and llama.cpp-Metal
or MLX. With greedy decoding the difference is usually <1pp on
benchmarks like MMLU-Redux, but it is non-zero. Comparisons to
PrismML's *absolute* numbers carry that drift; comparisons of two
models within your own harness are clean.

### 2. The Gemini fallback judge

The rule-based extractor handles most answers; Gemini 2.5 Flash Lite
is invoked only when the rule parser fails. Two paths:

- **Get a Gemini API key**, drop into EvalScope's judge config. Cost
  is negligible (Flash Lite, mostly thousands of tokens / benchmark
  run). This reproduces their setup faithfully.
- **Swap to a local judge LLM** (Qwen3-4B-Instruct, Llama-3.1-8B,
  whatever fits) at temp=0. Score is no longer directly comparable
  to PrismML's published numbers — but the *delta* between two
  models scored under the same local judge is meaningful. Track
  rule-parser-fallback rate; if <5% of items invoke the judge,
  judge-choice barely moves the score.

The judge invocation rate matters most for MMLU-Redux, GPQA, MATH-500,
MuSR. For IFEval/IFBench/BFCL/code benchmarks the judge is unused.

## Recommended local play

For testing whether a reproduction recipe matches Bonsai's
behavioural fingerprint, the cheapest meaningful test is:

1. Stand up `llama.cpp` server with Metal on M4 Max.
2. Run EvalScope against:
   - `Qwen3-8B` FP16 GGUF (the upstream baseline)
   - `Bonsai-8B-Q1_0.gguf` (PrismML's deployed 1-bit)
   - Your reproduction's deployed weights (Q1_0 GGUF)
3. Use a local judge model for fallback (or Gemini if you have a
   key).
4. Compare *gaps*: the Bonsai-8B → Qwen3-8B gap is 8.8 avg-points
   (70.5 → 79.3) under PrismML's setup. Reproducing that gap (with
   appropriate engine drift) is a cleaner success criterion than
   landing on the absolute 70.5.

## Memory footprint on a 36GB M4 Max

```
asset                              size (resident)
---------------------------------- ----------------
Qwen3-8B FP16 GGUF                 ~16.4 GB
Bonsai-8B Q1_0 GGUF                ~1.15 GB
Reproduction Q1_0 GGUF (similar)   ~1.15 GB
EvalScope harness                  ~1-2 GB
Local judge (4B FP16)              ~7-8 GB

worst-case load (Qwen3 + judge)    ~24-26 GB resident
```

Both Bonsai and Qwen3 fit alongside a 4B-class judge. If memory is
tight, swap one model in/out per benchmark sweep — `llama.cpp` is
fast to mmap-load.

## What NOT to bother replicating

- **The Pareto / intelligence-density chart**. It's downstream of the
  benchmark scores; if those are reproduced, the chart is just plot
  arithmetic.
- **The cross-platform throughput / energy tables**. Hardware-specific
  measurements; only the eval framework matters for recipe-validation.
- **VLLM_BATCH_INVARIANT** / **FA2 deterministic on H100**. On Apple
  Silicon the determinism story is engine-specific. Greedy decoding
  with seed 42 in `llama.cpp` is reproducible run-to-run on the same
  hardware; that's what matters for your own A/B testing.

## Summary

Replicate PrismML's benchmarks on M4 Max via:
- EvalScope (OSS) +
- llama.cpp Metal or MLX as the engine (instead of vLLM CUDA) +
- Gemini API key OR local-judge swap

Expected score drift vs PrismML's published numbers: small (~1pp) but
non-zero. For comparing a reproduction recipe against Bonsai itself,
that drift is a constant offset and the comparison stays clean.
