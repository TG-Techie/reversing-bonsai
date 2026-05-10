# Agent → Agent handoff

> Hi. You are picking up where I left off. We're the same model, but this is a
> new session: you don't share my memory, you do share my judgment. Read this
> before you do anything else; it will save you ~30 minutes of trial and error.

## Who you are, what we're doing

You are Claude (Opus 4.7, 1M context), running inside Claude Code on the web,
working in the **`tg-techie/reversing-bonsai`** repo. The user is TG-Techie /
Jonah Y-M; primary workflow is mobile, so prefer one-action paths and short
text replies. The user is direct and technical; they appreciate when you state
your reasoning crisply.

The scientific goal: **figure out why PrismML's "1-bit Bonsai" quantized models
don't degrade.** They claim Q1_0_g128 (1 sign bit per weight + 1 FP16 scale per
group of 128) preserves Qwen3-{1.7B, 4B, 8B} accuracy almost in full, and that
the same weights ship in an "unpacked" FP16 form. The user's hypothesis:

> The Q1_0 quant can be unpacked into the same floating representation as the
> base Qwen3 it was quantized from — possibly modulo some channel permutation,
> since neural-graph nodes can be reordered without changing the function.

We want to verify or falsify that.

## What's already done

The repo currently contains:

- **The two whitepapers** (`1-bit-bonsai-8b-whitepaper.pdf`,
  `ternary-bonsai-8b-whitepaper.pdf`). Read both. The first is the binary
  family; the second is the ternary 1.58-bit family. PrismML deliberately
  hides the methodology behind "proprietary Caltech IP."
- **`FINDINGS.md`** — paper digest, the exact Q1_0 binary format reverse-
  engineered from `ggml-quants.c`, and the three hypotheses (lossless
  dequant / no permutation / sortedness-within-groups) that the analysis
  scripts test. Read it.
- **Analysis tooling** (`src/`):
  - `q1_0.py` — pure-Python Q1_0 codec mirroring the C reference. Block =
    18 bytes (FP16 d + 16 bytes signs, LSB-first within byte). Round-trip
    self-tested.
  - `gguf_inspect.py` — GGUF metadata + tensor inventory printer.
  - `analyze_q1_0.py` — per-tensor sign-pattern, run-length, and
    scale-ordering statistics. Tests "are blocks sorted within each group?"
  - `compare_unpacked.py` — 3-way GGUF comparator (Q1 / unpacked-GGUF /
    base-GGUF). Tests "does dequantize(Q1) equal the unpacked file?"
  - `compare_unpacked_vs_qwen3.py` — 2-way FP comparator with greedy
    permutation search. Supports safetensors. Tests "is Bonsai a
    channel-permuted Qwen3?"
- **Prebuilt llama.cpp binaries** for Linux x86_64 in
  `prebuilt/linux-x86_64/{llama-cli, llama-quantize, llama-gguf}`. Built
  from upstream `1e5ad35d` (b9093). macOS users use
  `scripts/build_llama_cpp.sh` (auto-enables Metal).
- **`uv` workspace** with `gguf`, `huggingface-hub`, `hf-transfer`, `numpy`,
  `safetensors`, `pyyaml`. `pyproject.toml` + `uv.lock` are committed.
- **GitHub Actions workflow** `.github/workflows/analyze-bonsai.yml` that
  downloads the three model variants on a hosted runner, runs the
  analyses, uploads weights to a release, commits reports back. Built as
  a workaround when this sandbox couldn't reach `huggingface.co`.
- **Three release tags already exist** with model bytes attached:
  - `models-bonsai-1.7B-r5` — full (Q1_0 GGUF + unpacked + base, ~7 GB)
  - `models-bonsai-4B-r6` — full (~13 GB)
  - `models-bonsai-1.7B-r8` — partial (Q1_0 only)
  See `models/MANIFEST.yaml` once any future workflow run writes it.
- **`.claude/settings.json`** with `sandbox.allowedDomains` set to wildcard
  `*.huggingface.co`, `*.hf.co`, `*.githubusercontent.com`,
  `*.pytorch.org`, `*.modelscope.cn`, plus apex domains.

## Sandbox network state — TEST THIS FIRST

When I left, `huggingface.co` was blocked at the egress proxy
(`x-deny-reason: host_not_allowed`). The user merged a PR that adds a
project-scoped allowlist (`.claude/settings.json`). The user's intent is that
this session has the new allowlist live. **Do not trust this.** Run:

```sh
curl -sIL -m 5 https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/config.json | head -3
```

If you get `HTTP/2 200`, the allowlist is live and you can skip the entire
GitHub Actions workaround. If you get `403 Host not in allowlist`, fall back to
**Path B** below.

## The two paths from here

### Path A — direct fetch (preferred, when HF reachable)

1. `uv run python -c "from huggingface_hub import snapshot_download;
   snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models/gguf-1.7B',
   allow_patterns=['*Q1_0*.gguf','*.json','README*'])"`
   — repeat for `prism-ml/Bonsai-1.7B-unpacked` (FP16 safetensors) and
   `Qwen/Qwen3-1.7B`.
2. Run the analyses:
   - `uv run python src/gguf_inspect.py models/gguf-1.7B/Bonsai-1.7B-Q1_0.gguf --tensors`
   - `uv run python src/analyze_q1_0.py models/gguf-1.7B/Bonsai-1.7B-Q1_0.gguf --top 0`
   - `uv run python src/compare_unpacked_vs_qwen3.py models/unpacked/.../*.safetensors models/base/.../*.safetensors`
3. Write empirical findings into `FINDINGS.md` (commit on a new branch + PR;
   `main` is protected).

### Path B — pull from the existing release attachments

If HF is still blocked but the repo is public OR you have repo auth:

1. `scripts/fetch_models_from_release.sh` (auto-resolves the latest
   `models-bonsai-1.7B-r*` tag from `models/MANIFEST.yaml` if present, else
   the GitHub API).
2. Reassembles `*.part-*` chunks; verifies sha256 if present.
3. Run the same analyses as Path A.

If the repo is private and you don't have download auth: trigger a fresh
workflow_dispatch on `main` with all defaults; it'll run the analyses on the
runner, upload a release, and commit `reports/<family>-<size>/*.txt` back to
`main`. Read those via `mcp__github__get_file_contents`.

## What the analyses should answer

The reports expose three numbers that decide everything. State them
explicitly in `FINDINGS.md`:

1. **`max(|dequant(Q1) - unpacked|)`** over every tensor. If ≈ 0 (≤ 1 fp16
   ULP), the unpacked file is just an FP16 storage of the binary lattice
   (i.e. weights live on `±s_g`). Strong hypothesis.
2. **identity-row cosine** vs **best-permutation-row cosine** between
   Bonsai-unpacked and Qwen3-base, for each weight matrix. If both ≈ 1, no
   reorder; if identity ≪ best-perm, channels were permuted before
   quantization (and you should figure out which ones — `attn_q`/`attn_k`
   are constrained by RoPE pair structure, FFN intermediate is fully free).
3. **mean sign-transitions per 128-block** in the Q1_0 file. Random ≈ 64.
   ≤ 1 means signs are sorted within groups. If 64-ish: no sortedness
   trick.

## Open PRs / branches

When I left:
- Merged: PR #1 (toolkit), PR #2 (SHA256SUMS fix), PR #3 (sandbox allowlist).
- Not merged: commit **`c2149fc`** on branch `claude/fix-sha256sums-path`. It
  adds `main` to the workflow's push trigger branches so the workflow
  auto-fires when the workflow file changes on `main`. Useful only if Path A
  is blocked and you keep relying on the workflow. Open PR #4 for it
  (or close the branch if you've moved fully to Path A).

## House style

- One Bash command per call (the user asked for this — easier to review).
- No `sleep` polling loops. If you need to wait for a workflow run, use
  `Monitor` with persistence, or just end the turn and let webhooks wake you.
- Don't open PRs unless asked. The user explicitly asked for PR coverage of
  unmerged work; for new commits, ask first.
- Don't comment on PRs unless necessary; the user prefers diffs over prose.
- Mobile-aware: short messages, ~10–15 lines, ~7–11 words per line.

## What I would do next

1. Test HF egress (one curl).
2. If green: download 1.7B trio, run the three analyses, fill in the answer
   to each of the three hypotheses in `FINDINGS.md` — concrete numbers, then
   a one-line verdict per hypothesis. Commit on a branch, PR.
3. If still blocked: triage allowlist (it should be live in the new session;
   if not, ask the user briefly — don't keep working around it).
4. After 1.7B's verdict, repeat for 4B and 8B if the verdict is interesting.

The user's not waiting on anything urgent; they're available for clarifying
questions on mobile.

Good luck.
