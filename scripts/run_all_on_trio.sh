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

echo "== 03 Bonsai-unpacked vs Qwen3-base, layer 0 / mid / last (H2) =="
{
  for f in "model.layers.0\\." "model.layers.7\\." "model.layers.13\\." \
           "model.layers.20\\." "model.layers.27\\." "model.layers.35\\."; do
    echo "===== filter: $f ====="
    # --filter takes a substring; we strip the trailing escape for the regex
    # (the script does plain-substring matching, not regex).
    uv run python src/compare_unpacked_vs_qwen3.py \
        "$UNPACKED" "$BASE" \
        --filter "${f%\\.}." 2>&1 | tail -90 || true
  done
} > "$DEST/03_unpacked_vs_qwen3.txt"

echo "== 05 dequant(Q1_0) vs unpacked safetensors (H1) =="
uv run python src/compare_q1_dequant_vs_unpacked.py \
    "$GGUF" "$UNPACKED" > "$DEST/05_dequant_vs_unpacked.txt"

echo "== 06 magnitude follow-up (per-block, per-row, per-col) =="
{
  for f in "blk.0\\." "blk.7\\." "blk.13\\." "blk.20\\." "blk.27\\." "blk.35\\."; do
    echo "===== filter regex: $f ====="
    uv run python src/compare_magnitudes.py \
        "$GGUF" "$UNPACKED" "$BASE" \
        --filter "${f%\\.}." 2>&1 | tail -90 || true
  done
} > "$DEST/06_magnitudes.txt"

echo "== 07 input-column permutation test (H4) =="
{
  for f in "model.layers.0\\." "model.layers.13\\." "model.layers.27\\." "model.layers.35\\."; do
    echo "===== filter: $f ====="
    uv run python src/test_column_permutation.py \
        "$UNPACKED" "$BASE" \
        --filter "${f%\\.}." 2>&1 | tail -25 || true
  done
} > "$DEST/07_col_permutation.txt"

echo "[done] reports under $DEST"
ls -lh "$DEST/"
