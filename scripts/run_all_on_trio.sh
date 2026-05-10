#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
#
# Run all analysis scripts unchanged on a staged 1.7B / 4B / 8B trio.
# Mirrors the workflow in .github/workflows/analyze-bonsai.yml and serves as
# a "blind second case" runner: pass the SAME script set the 1.7B run used,
# without parameter tweaking, so cross-size comparisons aren't biased by
# our 1.7B findings.
#
# Usage:
#   scripts/run_all_on_trio.sh <gguf> <unpacked.safetensors> <base.safetensors> <reports_dir>
#
# Example:
#   scripts/run_all_on_trio.sh \
#       models/q1/Bonsai-4B-Q1_0.gguf \
#       models/unpacked/model-00001-of-00002.safetensors \
#       models/base/model-00001-of-00003.safetensors \
#       reports/bonsai-4B

set -euo pipefail

GGUF="${1:?gguf path}"
UNPACKED="${2:?unpacked safetensors path}"
BASE="${3:?base safetensors path}"
DEST="${4:?reports dir}"

mkdir -p "$DEST"

echo "== 01 GGUF metadata + tensor inventory =="
uv run python src/gguf_inspect.py "$GGUF" --tensors > "$DEST/01_metadata.txt"

echo "== 02 Q1_0 sortedness / sign-pattern (H3) =="
uv run python src/analyze_q1_0.py "$GGUF" --top 0 > "$DEST/02_q1_0_analysis.txt"

echo "== Determine layer count from GGUF metadata =="
N_LAYERS=$(uv run python -c "
from gguf import GGUFReader
r = GGUFReader('$GGUF', 'r')
for k, f in r.fields.items():
    if k.endswith('.block_count'):
        print(int(f.contents()))
        break
" 2>/dev/null)
[ -z "$N_LAYERS" ] && N_LAYERS=28
echo "N_LAYERS=$N_LAYERS"

echo "== 03 Bonsai-unpacked vs Qwen3-base, EVERY transformer block (H2) =="
{
  for i in $(seq 0 $((N_LAYERS - 1))); do
    echo "===== filter: model.layers.${i}.self_attn ====="
    uv run python src/compare_unpacked_vs_qwen3.py \
        "$UNPACKED" "$BASE" \
        --filter "model.layers.${i}.self_attn" 2>&1 | tail -60 || true
    echo "===== filter: model.layers.${i}.mlp ====="
    uv run python src/compare_unpacked_vs_qwen3.py \
        "$UNPACKED" "$BASE" \
        --filter "model.layers.${i}.mlp" 2>&1 | tail -40 || true
  done
} > "$DEST/03_unpacked_vs_qwen3.txt"

echo "== 05 dequant(Q1_0) vs unpacked safetensors (H1) =="
uv run python src/compare_q1_dequant_vs_unpacked.py \
    "$GGUF" "$UNPACKED" > "$DEST/05_dequant_vs_unpacked.txt"

echo "== 06 magnitude follow-up (per-block, per-row, per-col), EVERY block =="
{
  for i in $(seq 0 $((N_LAYERS - 1))); do
    echo "===== filter regex: blk.${i}. ====="
    uv run python src/compare_magnitudes.py \
        "$GGUF" "$UNPACKED" "$BASE" \
        --filter "blk.${i}." 2>&1 | tail -90 || true
  done
} > "$DEST/06_magnitudes.txt"

echo "== 07 input-column permutation test (H4), EVERY block =="
{
  for i in $(seq 0 $((N_LAYERS - 1))); do
    echo "===== filter: model.layers.${i}. ====="
    uv run python src/test_column_permutation.py \
        "$UNPACKED" "$BASE" \
        --filter "model.layers.${i}." 2>&1 | tail -25 || true
  done
} > "$DEST/07_col_permutation.txt"

echo "[done] reports under $DEST"
ls -lh "$DEST/"
