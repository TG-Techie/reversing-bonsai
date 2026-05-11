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
IP" but **does not say what that IP is**. It might be a training
procedure (QAT, distillation, calibration); it might be a graph
transformation that re-parameterises the network before quantisation;
it might be something further afield mathematically — an optimizer
formulation, a basis change, an iterative reconstruction algorithm.
The bytes don't directly tell us which.

This repo's job is to **read the deployed bytes carefully enough that
we can constrain what techniques are consistent with them, and rule
out the ones that aren't.** We are not retraining. We are not building
a fork. We are doing forensic reverse-engineering: every claim is
something the bytes attest to, and the constraints those claims put on
candidate techniques add up to a shrinking hypothesis space.

The audience for findings is whoever later tries to recreate Bonsai-
style compression on a different model. They need to know *what's
load-bearing about the format, and what the bytes prove the technique
must (or must not) include* — independent of any specific guess at
what that technique is. The discipline below is meant to keep us from
collapsing the open hypothesis space prematurely onto whatever
specific technique we happened to think of first.

## Why this is empirical science, not coding — and why the scientific method applies

A normal codebase rewards "I ran it, it worked, ship it." Here the
artifacts are the truth, the scripts are just lenses, and a sloppy
reading of a lens output produces a wrong constraint on the technique
that someone later spends compute trying to reproduce.

The discipline that does work here is **the scientific method** in
its actual form — pre-register a question, design an experiment that
could distinguish the relevant hypotheses, run it, report the result
with its uncertainty, and update the hypothesis space accordingly.
Not "let me run all the scripts and see what's interesting." That's
the data-mining failure mode and it's how you end up confidently
claiming patterns that were artifacts of how you sliced the data.

Concretely the common failure modes that the scientific method
guards against:

- **A script ran successfully → I have a finding.** The output is
  numbers; whether they say what you think requires you to specify
  *what was being compared, with what statistic, against what null*
  before you started. Otherwise the post-hoc story is whatever feels
  satisfying — which is the definition of confirmation bias.
- **Reading the tail and projecting a trend.** "Layer 35 is 30% and
  layer 0 was 21%, so it's monotonically rising" is true on average
  but if layers 4–15 are flat the implied technique is different (a
  middle-layer plateau suggests early/late regions get different
  treatment from whatever produced these weights). The mean hides
  the shape.
- **"Storage precision" mistaken for "having stayed at teacher
  values".** A tensor stored in F32 (vs the Q1_0 1-bit format) is
  *format-preserved*. It may still differ from the teacher — and the
  per-element diff vs the teacher tells us how much. Don't infer
  "unchanged" from "stored at higher precision".
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
keepalive *them* — but **you keepalive while one is running**, the
same way you do for any other async work. A spawned sub-agent is a
background process from the parent's perspective; if you end your
turn waiting for it, the harness can suspend you and the sub-agent's
result lands silently.

## Why keepalive (and announce it)

The CC-on-the-web harness suspends turns that have no foreground
work — so without a `sleep`, your own agent loop halts. If you kicked
off `nohup ... &` and ended your turn, the harness can pause the
session and your background process can finish without you ever
seeing the result. The pattern that works is:

1. Start the long-running work as a true background process
   (`nohup ... &`, **not** `run_in_background: true` on the harness
   — the harness wrapper has been observed to lose work across
   suspends).
2. Hold the turn open with a foreground `sleep` of duration matched
   to expected work-time. You have agency: pick from
   **30 / 60 / 90 / 180 s** sized to the work in flight, with **300 s
   a last resort**. Bracket the call with timestamps so the gap is
   visible: `date -u +%H:%M:%S && sleep N && date -u +%H:%M:%S`.
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

## Resuming from compaction or summary

If your context was compacted or you're resuming from a summarized
session, three things you owe the user before you do anything else:
**say so explicitly** so the transparency line isn't broken, **re-Read
this file** (CLAUDE.md), and **re-Read the two whitepapers**
(`1-bit-bonsai-8b-whitepaper.pdf` and `ternary-bonsai-8b-whitepaper.pdf`)
— a summary of any of these isn't the same thing as having them in
front of you, and the discipline rules above plus the whitepaper
specifics are exactly the kind of thing that quietly drift when you
only have a paraphrase.

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

## How long-running autonomous research goes here

When the user grants you a multi-hour time window with periodic
check-ins instead of turn-by-turn driving, the failure modes shift.
The discipline this section captures is what made the May 2026 5-hour
run productive (PR #7 + extending into recipe-extraction). Future
agents arriving cold into a similar grant should reason from these,
not follow them by rote.

**Re-state the goal in your own words at the start of every grant.**
The user will course-correct your phrasing if you've drifted. This
also forces you to reload the goal into your active context when it's
been a while since the last time you touched it. The single sentence
that captures it for this repo is something like *"reverse-engineer
how Bonsai compresses Qwen3 to 1.125 bits/weight at ~95% benchmark
retention, deeply enough that a different base model could be
compressed the same way; output is byte-attested constraints, not
guessed recipes."* If you can't articulate this, re-Read the
whitepapers and `RECIPE_HINTS.md` before doing anything else.

**Don't ask permission, pick one and continue.** When the user gives
you a multi-hour window they explicitly do not want to be the driver.
"Should I continue or wind down?" "Should I do A or B?" — these halt
their ability to do other things. Pick the higher-leverage option
and act; if the user disagrees they will course-correct.

**In autonomous-grant mode, the ONLY reason to stop is the budget
running out.** If you finish a batch and find yourself about to ask
"which direction next?" — don't. Pick one (or more in parallel) and
execute. The user can always redirect mid-flight; you cannot recover
from minutes spent waiting for them. Asking-for-direction at
batch-boundaries is the most common failure mode for a grant; if
you're tempted, re-read this paragraph. Halting on binary questions
wastes the grant. This is not the same as silent-action — see
"communication pacing" below.

**Communication pacing: concrete findings, not narration.** The user
is on mobile, often watching the transcript stream while doing other
things. Before each tool call write one sentence of intent; after a
batch of work write a short status with the actual numbers and what
they mean. Avoid "I'm now going to..." style narration — describe
what changed, what's open, and ask only the questions the user can
actually answer.

**Sense when something is taking too long and kill it.** If a script
has been running for more than 2× its expected wall time, or has
produced no output for many minutes when it should be streaming,
investigate root cause and rewrite. Do not let it consume the budget.
The 36-layer audit in this run hit this — first version walked the
shard index 254 times (each tensor lookup is O(shards)); second
version walks the index 5 times. Same logic, ~20× faster. The win
came from killing-and-rewriting at the right moment, not from
patience.

**Always check at least 5–7 instances, never just 3.** A trio is
the minimum for early/mid/late framing, but with only 3 points you
can't see non-monotonic patterns and you'll falsely report "monotonic
trend" when the truth is "U-shape with peak in middle, gradual rise
late". The 4B sign-disagreement number in this run got reported as
"monotonic 21% → 30%" until the user pushed back and a 36-layer
re-read showed it was actually plateau-then-late-rise. 5–7 layers
spread across depth catches this.

**Periodic concrete-finding check-ins.** Roughly every 20–40 minutes
of clock time, post a short message to the user: "since last check
I added X, Y, Z results; here are the headline numbers; here's
what's open." This is the pacing that lets the user re-direct
without you halting. They don't need the running narrative; they
need the deltas.

**Sub-agents are for independent verification, not for delegating
work to.** Spawning a worktree-isolated sub-agent with a tight prompt
is the cheapest way to surface assumptions you slid past. Brief them
like a smart colleague who walked into the room cold: state the
claim, hand over the data path, ask for the numbers they themselves
measured. Bound the report length. **Do not let sub-agents see your
prior framing.** Two valuable patterns from this run:

1. *Observation-only narrative derivation.* Hand the agent only the
   §1 force-by-data observations (numbered list); ask them to derive
   what hypothesis space is consistent. They will surface alternative
   readings you missed. (See `reports/HYPOTHESIS_SPACE.md` — sub-agent
   produced H_a/H_b/H_c, of which H_c was falsified by data the
   author already had but hadn't quite framed as a falsifier.)
2. *Prior-art research with a tight scope.* Spawn one for "broad
   literature in topic X" and a separate one for "the specific people
   behind this artifact and their own publications". Do not combine —
   the search strategies are different and the second is much more
   constrained. (See `reports/RELATED_RESEARCH.md` for output.)
3. *Recipe-as-written verification.* When you have a candidate recipe
   you've spent hours building up, spawn a sub-agent and give it the
   recipe + the data path. Tell it to *independently measure* the
   recipe's claims and report numbers. **Specifically tell it your
   known biases** ("the author has a known confirmation bias toward
   X; if you see evidence inconsistent with X, point it out"). This
   is the pattern that caught real over-statements in this run —
   each catch came from the same loop:
   - **Round 1**: the magnitude-graded sign-flip pattern was read
     as evidence FOR a LoRA preprocess, when in fact it's consistent
     with several mechanisms. Corrected in `RECIPE_HINTS.md` v4.
   - **Round 2**: an "SGD-α step is concentrated at deep MLP gate/up"
     claim was a Gaussian-sigma-misspecification artifact (the
     simulator used 1.25× the wrong tensor's ratio uniformly).
     Retracted in `reports/local-8B/36_*`.
   - **Round 3**: a "within-block attn-before-MLP ordering" claim
     was over-stated as the byte-attested mechanism when several
     alternatives (SwiGLU-sensitivity-weighted parallel pass,
     MLP-only small-rank LoRA, calibration-data composition,
     layer-norm asymmetry) produce the same byte signatures, AND
     the proposed mechanism's direction-of-gap prediction failed
     at L0 (8B) and L1 (1.7B). Corrected in `reports/local-8B/44_*`.
   - **Round 4**: an extension of the LoRA-rank simulation found
     that the L1-3 MLP over-dispersion spike is partially explained
     by Qwen3-8B's intrinsic teacher block-magnitude heterogeneity
     (~10× higher block-CV at L1-3 gate vs other depths). The
     "recipe makes block-coherent decisions at L1-3" framing was
     weakened — most of the spike comes from teacher structure that
     ANY perturbation would surface. Documented in
     `reports/local-8B/45_*`.
   - **Round 5**: an over-attribution within `45_*` claimed the
     recipe REDUCES the teacher's natural over-dispersion via
     per-block scale tuning. Verifier-5 identified that the
     comparison simulation doesn't include per-block scaling, so
     the direction of recipe contribution (homogenising vs
     amplifying) is not byte-attested. Also flagged that SUMMARY's
     "approximately full-rank delta" framing undersells what
     rank-16 fractions actually say (small-rank LoRA NOT ruled out),
     and v_proj's depth-rise was not teacher-confound-checked.
     Corrected in `45_*`, `SUMMARY.md` findings 18, 20, 22.

   The pattern is robust: when you've been organising-around a
   hypothesis for a while, every byte signature looks like
   confirmation. A fresh-context sub-agent with explicit
   bias-naming reliably surfaces over-reaches. **Use this pattern
   before publishing or merging recipe-style claims.** The
   verifier doesn't need to be right about every alternative it
   raises; it just needs to identify ONE that you didn't rule out
   to deflate an over-strong claim.

**Compare against the deterministic formula AND the base model
explicitly; sync state in messages.** The "formula" `sign(w_base) ·
mean(|w_base|_per_128_block)` is by definition the naive Q1_0
quantisation of the teacher; that's the same as comparing against
"naive-quant(base)". When you say "Bonsai vs base" you might mean
either of these — be explicit which. If you want to disentangle
"sign chosen well" from "scale chosen well" you need both
comparisons.

**Block alignment is exact.** Both Bonsai (post-HF-reverse) and base
reshape `(out, n_blk, 128)` row-major and the i-th element at
`(row, blk, i)` corresponds. Stating this once in each new analysis
script keeps it from drifting; "block group to precise weight
placement" is the user's phrase and it matters because comparing
*group statistics* of misaligned groups is meaningless even if the
numbers look reasonable.

**Confounder discipline.** Some observations confound others. The
identity-shaping step ("Bonsai/PrismML/Caltech" emerges from temp=0
inference with empty system prompt) is force-by-data, but it is also
a *confounder for every byte-level cross-size or per-tensor claim*
because we cannot apportion drift between "the quantisation step"
and "the identity step". Flag confounders in `ASSUMPTIONS_AUDIT.md`
and re-flag them when you cite the affected observations.

**Pointers > bulk downloads for prior art.** When research is needed,
the right output is an arXiv URL list with one-sentence summaries of
which observation each paper would and wouldn't predict. Do not
download PDFs unless the user asks. The maintainer keeps the cache
clean; you write `RELATED_RESEARCH.md` with stable links, and when
PDFs land in `reports/related_papers/` they're added to that doc's
"in cache" list with the same metadata.

**Read the cited prior art, don't just summarise its abstract.** The
abstract gave us "ℓ∞-regularised → 1-bit"; reading the theorems and
corollaries gave us "boundary value `δ/λ` is *data-dependent, not
teacher-weight-dependent*", which is the differential prediction
that distinguishes ℓ∞-style recipes from "apply formula to teacher
and call it done". The discipline only works if you actually read the
prior work; paraphrase-from-memory will quietly converge on whatever
intuition you already had.

**Format: markdown tables only, each row on its own line, blank
lines around the table.** HTML tables don't render in this chat
interface, and tightly-packed markdown rows can render as one line.
Always test mentally: each row needs a leading newline before its
`|`.

**Branch per merge.** When the maintainer merges a PR, immediately
check out the new main and start a fresh branch for the next batch
of work. Don't add to the merged branch — it pollutes the main-line
history. The fresh-branch convention also gives you a clean
"as-of-merge" snapshot to point at later.

**Whitepapers compact fastest.** The two PDFs (`1-bit-bonsai-8b-
whitepaper.pdf`, `ternary-bonsai-8b-whitepaper.pdf`) are read once
at session start and rarely re-read; their specifics drift to
paraphrase as the context grows. If you find yourself struggling to
recall a specific claim from the whitepaper, re-Read the relevant
pages explicitly with `pages: "N-M"` (large PDFs require the page
parameter) before relying on memory.

## Why all of this is in CLAUDE.md and not a comment in some script

Because future agents arrive cold. They need the model of *what we're
trying to learn and why this discipline is the right discipline*
before they can usefully run the next experiment. The rules above are
illustrations; what we actually want is for the next agent to be able
to derive them from the situation when our specific examples don't
apply. Reason about the problem space first; the rules will follow.
