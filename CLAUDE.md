# Working notes for agents in this repo

> You are a fresh agent reading this with no prior context. The goal of
> these notes is to **transfer the *why*** behind how we work here, not
> to give you a checklist. Reason about the underlying problem; the
> rules are illustrations of what reasoning produces.

## What this repo is actually doing

PrismML released "1-bit Bonsai" — a Qwen3 child model where every
matrix-heavy weight has been quantized to 1 sign bit + a shared FP16
scale per group of 128 weights (the `Q1_0_g128` format). The
whitepaper attributes the quality preservation to "proprietary Caltech
IP" but never describes the training procedure.

This repo's job is to **read the deployed bytes carefully enough that
we can constrain what that procedure must have looked like.** We are
not retraining. We are not building a fork. We are doing forensic
reverse-engineering: every claim is something the bytes attest to,
and the constraints those claims put on a hypothetical reproduction
add up to a recipe sketch.

The audience for findings is whoever later tries to recreate
Bonsai-style compression on a different model. They need to know
*what's load-bearing about the format vs. the recipe*, because if you
copy the format and miss the recipe you get post-training-quant noise,
and if you copy the recipe and miss a format constraint you get an
inference-time failure. So the discipline below isn't aesthetic — it's
to keep that distinction sharp.

## Why this is empirical science, not coding

A normal codebase rewards "I ran it, it worked, ship it." Here the
artifacts are the truth, the scripts are just lenses, and a sloppy
reading of a lens output produces a wrong recipe constraint that
someone later spends compute trying to reproduce. Concretely the
common failure modes:

- **A script ran successfully → I have a finding.** The output is
  numbers; whether they say what you think requires you to specify
  *what was being compared, with what statistic, against what null*
  before you started. Otherwise the post-hoc story is whatever feels
  satisfying — which is the definition of confirmation bias.
- **Reading the tail and projecting a trend.** "Layer 35 is 30% and
  layer 0 was 21%, so it's monotonically rising" is true on average
  but if layers 4–15 are flat the recipe story is different (a
  middle-layer plateau suggests early/late regions get different
  treatment). The mean hides the shape.
- **"Storage precision" mistaken for "trainability".** A tensor
  stored in F32 (vs the Q1_0 1-bit format) is *format-preserved*. It
  may still have been trained during QAT — and the diff vs the
  teacher tells us. Don't infer freezeness from format.
- **Comparing across sizes by remembered numbers.** A "row cosine
  0.45 at 4B vs 0.50 at 1.7B" is meaningful only if the same script
  with the same arguments produced both. Mixing your local 4B run
  with a 1.7B number from a doc that may have used a different
  layer-set or aggregation is a category error you won't notice
  unless you re-run the smaller size locally.

The mental move that helps is to *pre-register the experiment in one
sentence* — what tensor, what axis, what statistic, what null — and
then write the result with **uncertainty** (range across measured
items, n in the average). If you find yourself reaching for prose to
explain a number rather than a tighter statistic, that's a signal to
rerun with better methodology.

The repo's findings docs already separate **observed** (reproducible
from the bytes) from **inferred** (consistent with the data, not
proven), and tag each implication as **force-by-format**,
**force-by-data**, or **suggestion**. That separation is load-bearing
because the reproducer can rely on the first, must verify the second.

## Why use a sub-agent as judge, and how

Your own confirmation bias is real and you can't introspect past it.
A fresh sub-agent with no prior context, given the data and the
specific claim, is the cheapest way to surface assumptions you slid
past. Use the `general-purpose` agent type with `isolation:
"worktree"` — the worktree matters because:

- it gives the sub-agent its own copy of the repo so concurrent
  edits between you and it don't race,
- it forces the verification to be self-contained: the sub-agent has
  to reproduce results from the artifacts on disk, not lean on your
  framing.

Brief them like a smart colleague who walked into the room cold:
state the claim, hand over the data path, ask for *the numbers they
themselves measured* and a HOLDS / WEAKER / DOESN'T HOLD verdict with
caveats. Bound the report length. Don't treat their assessment as
authoritative — treat it as the second eye on the same data, which
is enough to catch most assumption slips.

Sub-agents *terminate*. They are checks, not collaborators. Don't
keepalive them, don't chain them.

## Why keepalive (and announce it)

The CC-on-the-web harness suspends turns that have no foreground
work. If you kicked off `nohup ... &` and ended your turn, the
harness can pause the session and your background process can finish
without you ever seeing the result. The pattern that works is:

1. Start the long-running work as a true background process
   (`nohup ... &`, **not** `run_in_background: true` on the harness
   — the harness wrapper has been observed to lose work across
   suspends).
2. Hold the turn open with a foreground `sleep` of duration matched
   to expected work-time (~60 s for typical analyses on this VM).
3. Check completion in a *separate* Bash call afterward.

The "**announce before sleeping**" part — saying "Doing a 60s
keepalive while H2 finishes" — is for the user (who's often on
mobile and watching the transcript stream) and for future-you reading
the transcript later. A bare `sleep 60` looks indistinguishable from
the agent dying. The announcement makes the intent legible.

If a single `sleep` is the only thing keeping the session alive,
that's a single point of failure. Where you can, also drop a
secondary signal — a status file you're watching, a webhook
subscription, a follow-up poll — so the harness can't strand you on
one bad sleep.

This *only* applies to the top-level agent. Sub-agents should run
their assigned task and exit; making them keepalive defeats the
short-lived-judge purpose.

## Why mind disk

`df -h /` here reports 252 GB total but only ~30 GB is actually
allocatable (the rest is reserved blocks). When you hit the cap, the
failure mode is **bash itself returning exit code 1 with no
output** — bash can't fork when the disk is full. That looks like the
agent is broken, but it's just disk pressure. Cleaning model files
restores normal operation.

The classic trap is `cat *.part-* > file.tmp && mv file.tmp file` —
the `.tmp` intermediate doubles peak transient usage. With ~16 GB of
chunks in flight and only ~15 GB headroom that's a guaranteed OOM.
Either reassemble in place (`cat *.part-* > file && rm *.part-*`)
or stream-and-delete (`for p in *.part-*; do cat "$p" >> file && rm
"$p"; done`).

Empirical sizes / speeds for keepalive sizing:
- 252 GB nominal, ~30 GB allocatable, ~5 GB used by uv's first sync.
- 1.7B trio ~7 GB, 4B trio ~16 GB, 8B trio ~33 GB. **8B does not fit
  alongside another size**; free one before pulling the next.
- Release-asset download ~50 MB/s. 1.9 GB chunk = 35–40 s. Full 4B
  trio = ~5–6 min net. Full 8B = ~10–12 min net.
- Add ~30 s/shard for cat-reassembly.
- uv first-run sync pulls a CUDA-flavoured torch (~5 GB) we don't
  use; cached after first sync.

## Quirks of the artifact layout

- Release `models-bonsai-{size}-r{N}` packs **both** Bonsai-unpacked
  shards and Qwen3-base shards into the same flat asset list. They
  share the same `model.safetensors.index.json` filename, which one
  set overwrites the other on download. Move them into `unpacked/`
  and `base/` subdirectories before running comparators.
- HuggingFace egress is blocked at the sandbox proxy
  (`x-deny-reason: host_not_allowed`). `release-assets.githubusercontent.com`
  is allowed. Plan downloads accordingly: prefer release-asset
  fetches via the workflows in `.github/workflows/` over direct HF
  pulls.
- The Bonsai-unpacked safetensors file carries no information beyond
  the GGUF — `dequantize_row_q1_0(GGUF) ≡ Bonsai-unpacked` to FP16
  storage precision (H1, validated at 1.7B and 4B). For new sizes
  you can download just the GGUF + base and derive the unpacked
  on-demand. The `fetch-bonsai-8b.yml` workflow uses this trick to
  fit inside the 60-min runner cap.
- `main` is branch-protected; pushes from runners go to `claude/**`
  branches.

## What the existing scripts do (one-liner each)

- `src/q1_0.py` — pure-Python Q1_0 codec; round-trips byte-identical
  vs `ggml-quants.c`.
- `src/gguf_inspect.py` — GGUF metadata + tensor inventory.
- `src/analyze_q1_0.py` — H3: per-block sign / scale statistics.
- `src/compare_q1_dequant_vs_unpacked.py` — H1 bridge: dequant(Q1_0)
  vs Bonsai-unpacked.
- `src/compare_unpacked_vs_qwen3.py` — H2: identity row cosine + greedy
  best-row-perm; multi-shard aware.
- `src/compare_magnitudes.py` — magnitude follow-up; per-block, per-row,
  per-col stats.
- `src/test_column_permutation.py` — H4: per-input-column statistics
  vs base.
- `src/sign_disagreement.py` — per-tensor flip rate vs Qwen3-base.
- `src/ptq_baseline_v2.py` — calibration: PTQ-quant Qwen3 directly,
  measure cos vs original.
- `src/joint_permutation_search.py` — joint cross-tensor residual-stream
  permutation search.
- `scripts/fetch_models_from_release.sh` — fetch + reassemble from a
  release tag.
- `scripts/independent_verify.py` — sample of how a verification
  sub-agent reproduced H1/H2/PTQ/norms claims independently.

## Why all of this is in CLAUDE.md and not a comment in some script

Because future agents arrive cold. They need the model of *what we're
trying to learn and why this discipline is the right discipline*
before they can usefully run the next experiment. The rules above are
illustrations; what we actually want is for the next agent to be able
to derive them from the situation when our specific examples don't
apply. Reason about the problem space first; the rules will follow.
