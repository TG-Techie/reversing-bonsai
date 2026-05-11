# Working notes for agents in this repo

> Fresh agent reading this with no prior context: the goal is to
> transfer the *why* behind how we work here, not give you a
> checklist. Reason about the problem; the rules are illustrations
> of what that reasoning produces. Operational reference (script
> list, disk sizes, artifact-layout quirks) is in `GLOSSARY.md`.

## What this repo is doing

PrismML released "1-bit Bonsai" — a Qwen3 child model where every
matrix-heavy weight is 1 sign bit + a shared FP16 scale per 128
weights (`Q1_0_g128`). The whitepaper credits "proprietary Caltech
IP" without saying what it is.

This repo's job: **read the deployed bytes carefully enough to
constrain what techniques are consistent with them, and rule out
the ones that aren't.** Forensic, not retraining. The audience is
whoever later tries to recreate Bonsai-style compression on a
different base — they need to know what's load-bearing in the
format and what the bytes prove the technique must (or must not)
include. The discipline below exists to keep the hypothesis space
from collapsing prematurely onto whatever technique we happened to
think of first.

## Empirical science, not coding

A normal codebase rewards "I ran it, it worked, ship it." Here the
artifacts are the truth, the scripts are lenses, and a sloppy
reading produces a wrong constraint that someone later spends real
compute trying to reproduce.

The discipline is the scientific method in its actual form:
pre-register what you're testing in one sentence (tensor, axis,
statistic, null), run it, report with uncertainty, update the
hypothesis space. The data-mining failure mode — run everything,
notice patterns — is how you end up confidently claiming artifacts
of how you sliced the data.

Common failure modes:

- **Script ran → I have a finding.** Numbers aren't findings until
  you've specified what was compared against what null *before*
  running.
- **Reading the tail.** Mean across 36 layers hides shape; report
  the distribution.
- **Storage precision ≠ unchanged.** A tensor in F32 is format-
  preserved; it may still differ from teacher.
- **Across-size by remembered numbers.** Re-run, don't paraphrase.

The findings docs separate **observed** (reproducible from bytes)
from **inferred** (consistent, not proven), and tag each as
**force-by-format**, **force-by-data**, or **suggestion**. The
reproducer relies on the first and verifies the second; collapsing
them is the load-bearing failure to avoid.

## Sub-agent as judge

Confirmation bias is invisible from inside. A fresh-context
sub-agent with no prior framing is the cheapest way to surface
assumptions you slid past. Use `general-purpose` with
`isolation: "worktree"` so the verification is self-contained on a
copy of the repo.

Brief them like a colleague who walked in cold: state the claim,
hand over the data path, ask for numbers they measured themselves
and a HOLDS / WEAKER / DOESN'T HOLD verdict. Bound the report. Treat
the verdict as a second eye, not authority — enough to catch most
slips.

Sub-agents terminate; they're checks, not collaborators. You
keepalive while they run; *they* don't.

## Keepalive

The CC-on-the-web harness suspends turns with no foreground work.
Background processes started before the suspend can finish silently
without you seeing the result.

1. Start long work as `nohup ... &`. The harness's
   `run_in_background: true` has been observed to lose work across
   suspends.
2. Hold the turn open with a foreground `sleep` sized to expected
   work-time: 30 / 60 / 90 / 180 s; 300 s last resort. Bracket
   with timestamps so the gap is visible.
3. Check completion in a separate Bash call.

Announce before sleeping ("Doing a 60s keepalive while H2
finishes"). A bare `sleep` looks like the agent died; the
announcement makes intent legible to the mobile reader.

If `sleep` is your only liveness signal it's a single point of
failure — drop a secondary (status file, webhook, follow-up poll).
Sub-agents don't keepalive; only the top-level agent does.

## Resuming from compaction

Three things you owe the user before any work: **say so
explicitly**, **re-Read this file**, **re-Read both whitepapers**
(`1-bit-bonsai-8b-whitepaper.pdf`,
`ternary-bonsai-8b-whitepaper.pdf`). Paraphrase-from-summary drifts;
the discipline rules and whitepaper specifics are exactly what
quietly drift.

## Disk pressure is silent

When the disk fills, bash exits 1 with empty stdout (can't fork)
— looks like the agent broke, is just disk pressure. Check
`df -h /` first when a command fails empty. Sizes in `GLOSSARY.md`.

## Long-running autonomous research

Multi-hour grants with periodic check-ins (not turn-by-turn driving)
shift the failure modes. What matters:

**Re-state the goal at every grant.** Forces a context reload;
lets the user catch drift. If you can't articulate it, re-Read the
whitepapers and `RECIPE_HINTS.md` first.

**Pick one direction and act.** Multi-hour grant means the user
isn't driving. "Should I do A or B?" wastes the grant. Pick
higher-leverage; they'll redirect if needed.

**Stop only when budget runs out.** Asking for direction at batch
boundaries is the most common grant failure. Pick again.

**Communication: deltas, not narration.** The user is on mobile.
One sentence of intent before tool calls; short numeric status
after batches. Skip "I'm now going to..." prose.

**Kill stalled work.** Script running 2× expected time or no
streaming output → investigate root cause and rewrite. The grant
is non-renewable.

**Check 5-7+ instances.** Three points can't distinguish monotonic
from U-shaped from plateau-then-late-rise.

**Concrete-finding check-ins.** Every 20-40 min: what was added,
headline numbers, open questions. Lets the user redirect without
halting you.

**Sub-agents are checks, not delegations.** Worktree-isolated,
tight prompt, ask for measured numbers. Three valuable patterns:

1. *Observation-only derivation.* Hand the agent only force-by-data
   observations; ask what hypothesis space is consistent. They'll
   surface readings you missed.
2. *Prior-art with tight scope.* Separate sub-agents for "broad
   literature in X" vs "the specific authors of this artifact."
   Search strategies differ.
3. *Recipe verification.* Recipe + data path + your known biases
   ("the author favors X; flag evidence inconsistent with it").
   Confirmation bias is invisible from inside; identifying ONE
   unruled-out alternative deflates an over-strong claim. **Use
   before publishing or merging recipe-style claims.**

**Compare against the formula AND the base.** The formula
`sign(w_base) · mean(|w_base|_per_128_block)` is naive Q1_0 quant
of the teacher. "Bonsai vs base" can mean either; be explicit.
Disentangling sign-chosen-well from scale-chosen-well requires
both.

**Block alignment is exact.** `(out, n_blk, 128)` row-major;
element `(r, b, i)` corresponds in Bonsai and base. Re-state per
script. Comparing group stats of misaligned groups is meaningless
even when the numbers look fine.

**Confounder discipline.** The identity-shaping step
(`Bonsai/PrismML/Caltech` from temp=0 + empty system prompt) is
force-by-data AND a confounder for byte-level cross-size /
per-tensor claims — we can't apportion drift between quantisation
and identity-tune. Flag in `ASSUMPTIONS_AUDIT.md`; re-flag at each
cite.

**Pointers > bulk downloads for prior art.** Output an arXiv URL
list with one-sentence "predicts / doesn't predict" summaries.
Don't download PDFs unless asked.

**Read the cited prior art.** Theorems and corollaries carry the
differential predictions; abstracts collapse to whatever intuition
you already had.

**Markdown tables only.** HTML doesn't render here; packed markdown
can collapse to one line. Each row on its own line, blank lines
around.

**Branch per merge.** When a PR merges, check out new main and
start a fresh branch. Don't keep adding to the merged branch.

**Whitepapers drift first.** Read at session start, rarely re-read;
specifics drift to paraphrase as context grows. If recall is fuzzy
on a specific claim, re-Read with `pages: "N-M"` before relying on
memory.

## Why this is in CLAUDE.md

Future agents arrive cold. They need the model of what we're trying
to learn and why this is the right discipline before they can
usefully run the next experiment. The rules are illustrations; the
goal is for the next agent to derive them from the situation when
our specific examples don't apply.
