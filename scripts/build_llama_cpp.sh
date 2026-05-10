#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
#
# Reproduce the prebuilt llama.cpp binaries.
#
# Defaults: CPU-only, no curl. Pass GGML_METAL=1 / GGML_CUDA=1 to override.
#
# Tested on:
#   - Ubuntu 24.04 x86_64       (Claude Code web sandbox)
#   - macOS 14+ Apple Silicon   (Metal-enabled)
#
# Usage:
#   scripts/build_llama_cpp.sh [WORKDIR=$HOME/work/llama.cpp]

set -euo pipefail

WORKDIR=${1:-$HOME/work/llama.cpp}
COMMIT=${LLAMA_CPP_COMMIT:-1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0}
TARGETS=${TARGETS:-"llama-quantize llama-cli llama-gguf"}

GGML_CUDA=${GGML_CUDA:-OFF}
GGML_METAL=${GGML_METAL:-OFF}
GGML_VULKAN=${GGML_VULKAN:-OFF}

# Auto-enable Metal on Apple Silicon if the user didn't set it.
if [ "$(uname -s)" = "Darwin" ] && [ "$GGML_METAL" = "OFF" ] && [ -z "${GGML_METAL_FORCE_OFF:-}" ]; then
  GGML_METAL=ON
  echo "[i] macOS detected — enabling Metal. Set GGML_METAL_FORCE_OFF=1 to disable."
fi

if [ ! -d "$WORKDIR" ]; then
  mkdir -p "$(dirname "$WORKDIR")"
  git clone https://github.com/ggml-org/llama.cpp.git "$WORKDIR"
fi

cd "$WORKDIR"
git fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git checkout "$COMMIT"

cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA="$GGML_CUDA" \
  -DGGML_METAL="$GGML_METAL" \
  -DGGML_VULKAN="$GGML_VULKAN" \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" --target $TARGETS

echo
echo "Built:"
for t in $TARGETS; do
  ls -lh "build/bin/$t" 2>/dev/null || true
done

# Suggest where to drop binaries to land in prebuilt/<platform>/
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) PLAT=linux-x86_64 ;;
  Darwin-arm64) PLAT=macos-arm64 ;;
  Darwin-x86_64) PLAT=macos-x86_64 ;;
  *) PLAT=$(uname -s | tr 'A-Z' 'a-z')-$(uname -m) ;;
esac
echo
echo "To install into the repo's prebuilt/ tree:"
echo "  mkdir -p prebuilt/$PLAT && cp build/bin/{llama-cli,llama-quantize,llama-gguf} prebuilt/$PLAT/"
