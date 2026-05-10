# Working notes for agents in this repo

> Living notes from agents reverse-engineering Bonsai. Read this first.
> Two halves: (1) discipline — *how* to do this work, since it's
> empirical science not coding; (2) sandbox quirks — concrete VM/env
> traps that aren't obvious from the surface.

## Scientific discipline (don't skip)

This repo is reverse-engineering a model artifact. That's empirical
science, and it's easy to slip into coding mode where a script run
becomes a "finding." A few rules to stay honest:

- **Pre-register what you're measuring.** Before running a script,
  state in one sentence: what tensor/layer/axis is being compared,
  with what statistic, against what null. Vague "compare Bonsai vs
  base" produces vague conclusions.
- **A finding is a claim about the bytes; pause and stress-test it
  before writing it down.** Useful prompts to ask yourself:
  *what assumption did I make? what would a skeptical reader ask? is
  the trend genuinely monotonic, or did I read the tail?*
- **Distinguish observation from inference, in your own head and on
  the page.** "the bytes show X" is observed; "Bonsai's recipe
  therefore did Y" is inferred. Don't blur them. Use the existing
  *force-by-format* / *force-by-data* / *suggestion* labels.
- **Match methodology when comparing across sizes.** A "row cosine
  0.45 at 4B vs 0.50 at 1.7B" is only meaningful if both numbers came
  from the same script with the same arguments on the same kind of
  tensor list. Re-run the smaller size locally rather than citing
  remembered numbers from a doc.
- **Spot-check, don't trust.** When comparing two arrays, eyeball a
  few rows by hand: shape, dtype, sample values. The most common bug
  here is a silent slice misalignment (vocab trim, GGUF shape
  reversal, BF16 vs FP16 cast).
- **Use a low-context judge for headline claims.** Spawn a fresh
  sub-agent with no prior context, give it the data and the claim,
  and let it independently sanity-check. Anything that survives that
  is more trustworthy than your own conviction.
- **Numbers without uncertainty are aspirations.** Report a range
  (min/max across the layers measured), not just a mean — and say
  *how many layers* the mean averages over.
- **"Storage precision ≠ training freezeness."** Just because a
  tensor is stored in F32 doesn't mean it was held fixed during QAT.
  Check value-level deviation before claiming "preserved verbatim."

If a session is short on time, write fewer findings and back each one
harder, instead of pushing more half-checked claims.

## Sandbox VM execution notes

> Empirical observations from running this repo's analyses inside a fresh
> Claude Code-on-the-web session, May 2026. Things that surprised me are
> here so the next agent can plan around them without rediscovering them.

## Disk

- `df -h /` reports **252 GB total**, but only roughly **30 GB is actually
  allocatable** (the rest is reserved blocks). Hitting the cap surfaces as
  `ENOSPC` in tools or, more confusingly, as `bash: exit code 1` with
  zero output — `bash` itself can't fork when the disk is full.
- The 1.7B trio is ~7 GB and is comfortable. The 4B trio is ~16 GB and
  fits with margin. The **8B trio at ~33 GB does not fit** alongside the
  uv venv (~5 GB after first sync); free one before pulling the next.
- `/tmp` is on the same filesystem, but typically only ~30 MB in use —
  it's not the squeeze. The squeeze is the models directory.

## Reassembling chunked release assets

`scripts/fetch_models_from_release.sh`'s reassembly step does
`cat *.part-* > <name>.tmp && mv <name>.tmp <name>` for atomicity. That
**doubles peak transient disk usage** for the duration of the cat, which
is what blew up the 4B fetch in this session — both the chunks and the
in-progress `.tmp` were on disk simultaneously. Two safer patterns:

1. Reassemble in place, deleting chunks before continuing: `cat *.part-* > <name> && rm *.part-*`. This is what I did once the script's first
   pass had OOM'd.
2. Stream & remove per chunk:
   `for p in *.part-*; do cat "$p" >> "$name" && rm "$p"; done`.
   Lower peak usage, no `.tmp`.

Either way the **`.tmp` intermediate is the trap**.

## Release-asset download speeds

Empirically ~**50 MB/s** from `release-assets.githubusercontent.com` on
this VM. Useful conversions for sizing keepalives:

- 1.9 GB chunk: **~35–40 s**
- 4B unpacked (7.6 GB across 3 chunks): **~150 s**
- 4B base (7.6 GB across 3 chunks): **~150 s**
- Full 4B trio (16 GB): **~5–6 min wall-clock net**
- Full 8B trio (33 GB): **~10–12 min wall-clock net**

Add ~30 s/shard for the cat-reassembly step. So a fresh 4B fetch is
realistically a **~7-minute** end-to-end operation.

## uv first-run sync

`uv run python ...` on a fresh checkout pulls the project's dependency
graph into a new `.venv/`. The current `pyproject.toml` brings in a
**CUDA-flavoured torch** (cudnn, cublas, cusparse, cufft, nccl, …) even
though this VM is CPU-only. That's about **5 GB of disk** spent on
binaries that never run. A future cleanup could pin a CPU-only torch
extras-set; not urgent, just a footprint observation.

After first sync, subsequent `uv run` calls use the cached venv and are
fast.

## Repo split for shared-folder release assets

Release `models-bonsai-4B-r11` (and equivalents) packs **both** the
unpacked Bonsai shards and the base Qwen3 shards into the same release
asset list. They land in the same directory and the
`model.safetensors.index.json` from one set overwrites the other.
The comparator scripts iterate every `*.safetensors` in a directory,
so dropping the shards into `unpacked/` and `base/` subdirectories
disambiguates them and lets `compare_*.py` do the right thing without
modification.

## Polling pattern that actually works

(Reconfirms the HANDOFF.) Use `nohup ... &` for true background work,
keep the conversation turn alive with a foreground `sleep`, and check
progress with separate Bash calls. **Announce the poll** before starting
the keepalive — the user values traceability.

`run_in_background: true` on the harness is fine for short-lived
foreground commands accidentally backgrounded, but the user explicitly
prefers `nohup` for long-running work to avoid the harness losing the
process across session suspends.
