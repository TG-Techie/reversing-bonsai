#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
#
# Inference smoke test: load a Bonsai-Q1_0 GGUF with the prebuilt
# llama-cli, generate a fixed completion, and confirm the model produces
# coherent text. Useful as a sanity check that the GGUF is intact and
# our reverse-engineered Q1_0 understanding is consistent with the
# kernel that actually decodes it.
#
# Usage:
#   scripts/inference_smoke_test.sh [models/q1/Bonsai-1.7B-Q1_0.gguf]

set -euo pipefail

GGUF="${1:-models/q1/Bonsai-4B-Q1_0.gguf}"
LLAMA_CLI="prebuilt/linux-x86_64/llama-cli"
DEST="reports/inference_smoke"

if [ ! -x "$LLAMA_CLI" ]; then
  echo "missing $LLAMA_CLI; build via scripts/build_llama_cpp.sh first." >&2
  exit 1
fi
if [ ! -f "$GGUF" ]; then
  echo "missing GGUF at $GGUF" >&2
  exit 1
fi

mkdir -p "$DEST"
LABEL=$(basename "$GGUF" .gguf)
OUT="$DEST/${LABEL}.txt"

PROMPTS=(
  "The capital of France is"
  "In Shakespeare's Hamlet, the protagonist's most famous soliloquy begins:"
  "Write a Python function that computes the factorial of a non-negative integer."
)

{
  echo "# inference_smoke_test  $(date -u +%FT%TZ)"
  echo "gguf:   $GGUF"
  echo "binary: $LLAMA_CLI"
  for p in "${PROMPTS[@]}"; do
    echo
    echo "## prompt: $p"
    echo
    "$LLAMA_CLI" -m "$GGUF" \
        -p "$p" \
        -n 128 \
        --temp 0.7 \
        --top-p 0.9 \
        --no-display-prompt \
        --no-warmup \
        2>/dev/null \
      || echo "[error] llama-cli exited non-zero"
  done
} | tee "$OUT"

echo "[done] saved -> $OUT"
