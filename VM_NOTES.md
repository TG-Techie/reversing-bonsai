# Sandbox VM execution notes

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
