# Agent → Agent handoff

> Hi. You are picking up where I left off. We're the same model, but this is a
> new session: you don't share my memory, you do share my judgment. Read this
> before you do anything else; it will save you ~30 minutes of re-derivation.

## Who you are, what we're doing

You are Claude (Opus 4.7, 1M context) running inside Claude Code on the web,
working in **`tg-techie/reversing-bonsai`** for TG-Techie / Jonah Y-M.
Primary workflow is mobile, so prefer one-action paths and short text replies.
The user is direct and technical; they appreciate when you state your
reasoning crisply.

The scientific goal: **understand how PrismML's "1-bit Bonsai" preserves
Qwen3 accuracy at 1.125 bits/weight, in service of eventually recreating the
same compression on additional models.** Recreation is downstream; rigorous
reverse engineering comes first.

## Current state of the investigation

Empirical work on the 1.7B trio is in [`FINDINGS.md`](./FINDINGS.md), with
acronyms in [`GLOSSARY.md`](./GLOSSARY.md) and recipe-relevant implications
in [`RECIPE_HINTS.md`](./RECIPE_HINTS.md). Headline numbers:

- **H1 — `dequant(Bonsai-Q1_0) ≡ Bonsai-unpacked` (FP16) elementwise**
  — confirmed across all 197 1.7B Q1\_0 tensors. Worst max-abs diff
  9.77e-4 (1 FP16 ULP). Sign agreement 100%. 99.9947% of all 13.4M
  groups have one distinct |w|. The "unpacked" file is just FP16
  storage of the binary lattice — no second high-precision track.
- **H2 — row permutation does not improve cosine to Qwen3-base**
  — greedy best-row-perm cosine matches identity to ±1e-3 across
  every layer (cos 0.43–0.60). Channel ordering is preserved.
- **H3 — signs within each 128-block are statistically random**
  — mean transitions/block 63.50 vs Binomial(127, 0.5) expectation
  63.5; lag-1 autocorrelation ≈ 0; pos-count balanced.
- **H4 — input columns are not a permutation of Qwen3's columns**
  — per-col Pearson 0.05–0.13 *and* per-col Spearman 0.03–0.13;
  top-10% loudest column overlap 12–16% vs 10% chance.
  *Caveat:* this is a per-tensor test. A graph-equivalent residual-stream
  permutation across many tensors would need a joint search;
  `src/joint_permutation_search.py` does that.
- **Magnitude follow-up:** Bonsai's per-block scale `s_g` correlates
  with Qwen3 group mean(|w|) at +0.65 early, dropping to +0.37 late.
  Per-output-row mean|w| correlation is preserved (~0.7); per-input-
  column mean|w| correlation is structurally erased (≤ 0.13). Bonsai
  is consistently 1.4–3× *louder* than Qwen3.

The synthesis: Bonsai-1.7B looks like Qwen3-1.7B with the matrix-heavy
weights *retrained* to live on a strict ±s\_g binary lattice, with channel
ordering, head structure, and FFN intermediate dim all preserved verbatim.
Norms and per-head q/k norms stay in higher precision. The "proprietary
Caltech IP" the paper alludes to is most likely the training recipe
(QAT/distillation), not a layout trick. Throughout the docs *observed*
facts and *inferred* claims are clearly separated; keep that convention.

## Repo layout

```
src/
  q1_0.py                              # pure-Python Q1_0 codec
  gguf_inspect.py                      # GGUF metadata + tensor inventory
  analyze_q1_0.py                      # H3 — sign-pattern / sortedness
  compare_unpacked.py                  # 3-way GGUF comparator
  compare_q1_dequant_vs_unpacked.py    # H1 — bridge: dequant(Q1) vs unpacked
  compare_unpacked_vs_qwen3.py         # H2 — FP comparator + greedy row-perm.
                                       #   Multi-shard aware (pass a dir).
  compare_magnitudes.py                # magnitude-distribution follow-up
  test_column_permutation.py           # H4 — input-col permutation test
  joint_permutation_search.py          # joint cross-tensor residual-stream
                                       #   permutation search (Hungarian)
  ptq_baseline.py                      # naive Q1_0(Qwen3) vs Bonsai vs base
  deep_dive.py                         # 7 sections (norms, vocab, lm_head,
                                       #   per-projection cos, per-head mag,
                                       #   outliers, scale-shape)
  qat_toy_demo.py                      # tiny char-level transformer with
                                       #   BitLinear modules; trains on CPU
  make_mini_report_figures.py          # parametrized on --size {1.7B,4B,8B}
scripts/
  build_llama_cpp.sh
  fetch_models_from_release.sh
  run_local_analysis.sh
  run_all_on_trio.sh                   # full sweep runner; calls all of the
                                       #   above against a (gguf, unpacked,
                                       #   base) trio; loops every layer
  merge_shards.py                      # multi-shard safetensors merger,
                                       #   self-verifies via 3-tensor
                                       #   round-trip equality
  inference_smoke_test.sh              # llama-cli sanity test
.github/workflows/
  analyze-bonsai.yml                   # original (1.7B/4B/8B switchable)
  analyze-bonsai-8b.yml                # hardcoded 8B; touch to retrigger
prebuilt/linux-x86_64/{llama-cli,llama-quantize,llama-gguf}
reports/bonsai-1.7B/                   # 01..XX text reports + figures/
                                       #   + MINI_REPORT.md
HANDOFF.md GLOSSARY.md FINDINGS.md README.md RECIPE_HINTS.md
```

## Things to know about the environment

- **HuggingFace egress is blocked** at the sandbox proxy
  (`x-deny-reason: host_not_allowed`). The `.claude/settings.json`
  allowlist exists but doesn't take effect mid-session and may not be
  applied at all in fresh sessions. Test before relying on it:
  ```sh
  curl -sIL -m 5 https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json | head -3
  ```
  If `403`, fall back to release-asset fetches via `github.com` (which
  *is* allowed since the repo is public).
- **VM disk is real but tight.** `df -h /` shows 252 GB total but
  only ~30 GB is actually allocatable (reserved blocks). The 8B trio
  at 33 GB does not fit alongside any other size; free one before
  pulling the next.
- **Repo is public.** Release-asset downloads via
  `release-assets.githubusercontent.com` work without auth.
- **`main` is branch-protected.** The workflow's "push reports back to
  triggering ref" step silently fails on `main`. Trigger from
  `claude/**` branches and reports will land on the triggering branch.
- **Compute budget.** This VM is CPU-only. Tens-of-GB safetensors
  comparisons run fine on CPU; 1.7B / 4B / 8B QAT training does *not*.
  `qat_toy_demo.py` is a 100K-param char-level model — trains in
  minutes on CPU and is the demonstrable scale.
- **Existing release tags:** `models-bonsai-1.7B-r{2,3,4,5,8,9,10,13}`,
  `models-bonsai-4B-r{6,11}`, `models-bonsai-8B-r1`. The `r5` (1.7B)
  and `r11` (4B) tags have full trios; the others are partial because
  the workflow's 60-min cap is hit before all the safetensors uploads
  finish on the larger sizes.

## Polling (the only pattern the user wants you to use)

When you need to wait on long-running work (model downloads, multi-minute
analyses, GH workflow runs, etc.), kick the work off as a **true
background process** with `nohup ... &` (**not** the harness's
`run_in_background: true` — that wraps the work in a way the harness can
lose if it suspends the session, and bg-waiter/`Monitor` patterns have
caused the VM filesystem to roll back when relying on them).

Then keep your conversation turn alive with a *foreground* sleep call,
and check completion in a **separate** Bash call:

```sh
# call 1: keep-alive (300 is just an example duration)
date -u +%H:%M:%S && sleep 300 && date -u +%H:%M:%S

# call 2 (separate Bash): check state
ls -lh reports/bonsai-4B/
```

Do **not** use the `Monitor` tool, do **not** use bg waiters via
`run_in_background: true`, do **not** use `sleep` inside the runner
itself, and do **not** end your turn assuming you'll be woken by a
webhook or `<task-notification>`.

This permission may be revoked if the user is actively chatting in real
time. If they say so, switch to short status-check Bash calls per turn
and let the user prompt the next check.

## Other house style

- One Bash command per call (the user asked for this — easier to review).
- **PRs are fine to open without asking.**
- Don't comment on PRs unless necessary; the user prefers diffs over prose.
- Mobile-aware: short messages, ~10–15 lines, ~7–11 words per line.
- Distinguish *observed* numbers (reproducible from the bytes) from
  *inferred* claims (consistent with the data but not directly
  attested). FINDINGS.md and the mini reports already do this — keep
  the convention.

## What I would do next

1. Re-fetch the 4B trio (`scripts/fetch_models_from_release.sh`
   pointed at `models-bonsai-4B-r11` works; or use the bg-curl pattern
   to parallelize the chunks). Run `scripts/run_all_on_trio.sh`
   against it using **directory paths** — the comparators accept a
   directory of shards transparently, no merging needed. Output to
   `reports/bonsai-4B/`.
2. Wait on the 8B workflow (latest `models-bonsai-8B-r1` is partial;
   touching `.github/workflows/analyze-bonsai-8b.yml` retriggers).
   Once the full trio uploads, free 4B from disk and pull 8B.
3. Generate updated mini reports + a cross-size synthesis showing
   that H1/H3/H4 reproduce at 4B and 8B (or finding the place they
   don't). The cross-size axis is what tells us the recipe is uniform.
4. Run the creative pieces: `inference_smoke_test.sh`,
   `qat_toy_demo.py`, and validate `joint_permutation_search.py`
   against a known-good case before reading too much into its
   negative result on the real data.

Good luck.
