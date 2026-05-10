# reversing-bonsai

A reverse-engineering toolkit for [PrismML's 1-bit Bonsai](https://prism-ml.com)
quantized language models — the family of Qwen3-{1.7B, 4B, 8B} checkpoints
PrismML ships at 1.125 bits/weight (`Q1_0_g128`) with claimed near-baseline
accuracy.

The whitepapers are in this repo
([1-bit](./1-bit-bonsai-8b-whitepaper.pdf),
[ternary](./ternary-bonsai-8b-whitepaper.pdf)) and attribute the breakthrough
to "proprietary Caltech IP." That black-box claim is the reason this repo
exists: we'd like an empirical answer to *what is the Bonsai weight space
actually doing?*

## Reading order for someone arriving cold

**5-minute TL;DR is in `SUMMARY.md`.** If you want full state in
~30 minutes, read in this order:

0. **`SUMMARY.md`** — 5-minute conclusions (force-by-data only,
   inferred separately, mechanisms ruled out vs consistent).
1. **`CLAUDE.md`** — framing of what we're doing and why; discipline
   rules for an agent picking up the work.
2. **`RECIPE_HINTS.md`** — what the bytes attest to about Bonsai's
   production pipeline. Updates v1 → v2 → v3 → v4 stacked; the v4
   section at the bottom is current. Source-anchored to specific
   reports under `reports/local-8B/`.
3. **`reports/REPRODUCTION_SKELETON.md`** — pseudocode-level sketch
   of a candidate recipe with falsifying tests.
4. **`reports/PRIOR_ART_VERDICT_MATRIX.md`** — 9 published 1-bit /
   sub-2-bit techniques scored against Bonsai's bytes. Only PTQ1.61
   broadly matches.
5. **`reports/HASSIBI_LINF_RECIPE_NOTE.md`** — detailed reading of
   the Hassibi-group ℓ∞ paper (`reports/related_papers/...2402.10474.pdf`)
   with its testable predictions mapped to Bonsai byte tests.
6. **`reports/ASSUMPTIONS_AUDIT.md`** — 14 numbered assumptions in
   our methodology; flags the identity-shaping step as a confounder
   for every byte-level claim.
7. **`reports/HYPOTHESIS_SPACE.md`** — the four hypotheses (H_a..H_d)
   about the production pipeline. H_c is falsified.
8. **`reports/CROSS_SIZE_AUDIT_SYNTHESIS.md`** — what holds across
   1.7B / 4B / 8B and what differs.
9. **`GLOSSARY.md`** — terminology reference.

The numbered reports under `reports/local-{1.7B,4B,8B}/` are the
raw byte-level measurements each conclusion is anchored to. The 8B
set is the most detailed. `reports/related_papers/` holds the PDFs
of prior-art techniques digested in (4).

## The questions we're trying to answer

Three falsifiable hypotheses, decided by three numbers:

1. **Lossless dequant** — does `dequantize(Bonsai-Q1_0)` equal `Bonsai-unpacked`
   (FP16) element-wise? If yes, the unpacked file is just FP16 storage of the
   binary lattice, i.e. all weights live exactly on `±s_g`.

2. **No permutation** — does `Bonsai-unpacked` match `Qwen3-{size}` row-by-row?
   If identity cosine ≈ 1, no channel reorder. If identity ≪ best-permutation
   cosine, channels were reordered before quantization (legal under GQA + RoPE
   constraints; FFN intermediate is fully free).

3. **Sign sortedness** — within each 128-element group, are signs clustered
   (≤1 transition) or random (~64 transitions)? Sortedness implies the
   permutation in (2) was chosen to make signs run-length-compressible.

See [`FINDINGS.md`](./FINDINGS.md) for the running set of empirical answers.

## Q1_0_g128 format (verified against `ggml-quants.c`)

```c
#define QK1_0 128
struct block_q1_0 {
    ggml_half d;             // FP16 scale = mean(|x|) over the 128 weights
    uint8_t   qs[QK1_0 / 8]; // 16 bytes -> 128 sign bits, LSB-first within byte
};                           // 18 bytes total
```

`w_i = s_g · (2·b_i − 1)` with `b_i ∈ {0, 1}`. Effective storage 1 + 16/128 =
**1.125 bits/weight**.

A pure-Python codec mirroring the C reference (round-trip self-tested) lives
at [`src/q1_0.py`](./src/q1_0.py).

## Toolkit

All scripts run under `uv run python`:

| Script | Purpose |
| --- | --- |
| `src/q1_0.py` | Q1_0 codec (parse + dequant + reference quantizer) |
| `src/gguf_inspect.py` | Print GGUF metadata + tensor inventory |
| `src/analyze_q1_0.py` | Per-tensor sign-pattern + scale-ordering stats (Hypothesis 3) |
| `src/compare_unpacked.py` | 3-way GGUF comparator (Q1 / unpacked-GGUF / base-GGUF) |
| `src/compare_q1_dequant_vs_unpacked.py` | dequant(Q1_0 GGUF) vs unpacked safetensors (Hypothesis 1) |
| `src/compare_unpacked_vs_qwen3.py` | FP comparator with greedy permutation search (Hypothesis 2) |
| `scripts/build_llama_cpp.sh` | Reproduce the prebuilt `llama-cli`/`llama-quantize`/`llama-gguf` |
| `scripts/fetch_models_from_release.sh` | Pull model artifacts from a release tag, reassemble chunks |
| `scripts/run_local_analysis.sh` | Run the full analysis pipeline against locally-staged models |

Prebuilt llama.cpp binaries (Linux x86_64) are in
[`prebuilt/linux-x86_64/`](./prebuilt/linux-x86_64/).

## How to get the models

### Option A — direct from HuggingFace (preferred when reachable)

```sh
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models/gguf-1.7B', \
                    allow_patterns=['*Q1_0*.gguf','*.json','README*'])"
```

Repeat for `prism-ml/Bonsai-1.7B-unpacked` (FP16 safetensors) and
`Qwen/Qwen3-1.7B`.

### Option B — from this repo's GitHub Releases

Each workflow run uploads the trio of models as a tagged release
(`models-bonsai-1.7B-r<n>`, `models-bonsai-4B-r<n>`, etc.) so the bytes are
fetchable without a HuggingFace round trip — useful for CI agents on
restricted networks.

```sh
scripts/fetch_models_from_release.sh                # latest 1.7B
SIZE=4B scripts/fetch_models_from_release.sh        # latest 4B
scripts/fetch_models_from_release.sh <tag>          # specific tag
```

### Option C — run it on a hosted runner

The workflow at [`.github/workflows/analyze-bonsai.yml`](./.github/workflows/analyze-bonsai.yml)
downloads the trio on a GitHub-hosted runner, runs every analysis, uploads the
models as a release, and commits the text reports to `reports/<family>-<size>/`.
Trigger manually from the **Actions** tab → *Analyze Bonsai Quantization* →
*Run workflow*.

## Running an analysis locally

```sh
uv sync
uv run python src/gguf_inspect.py models/gguf-1.7B/Bonsai-1.7B-Q1_0.gguf --tensors
uv run python src/analyze_q1_0.py models/gguf-1.7B/Bonsai-1.7B-Q1_0.gguf --top 0
uv run python src/compare_q1_dequant_vs_unpacked.py \
    models/gguf-1.7B/Bonsai-1.7B-Q1_0.gguf \
    models/unpacked-1.7B/model.safetensors
uv run python src/compare_unpacked_vs_qwen3.py \
    models/unpacked-1.7B/model.safetensors \
    models/base-1.7B/model.safetensors
```

## License

[MIT](./LICENSE) — Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie).

The Bonsai model files this toolkit analyzes are PrismML's, distributed by
PrismML under their own license terms. The Qwen3 base models are Alibaba's,
distributed under their own license terms. Nothing in this repo redistributes
those weights; the workflow only stages them transiently to produce the text
reports.
