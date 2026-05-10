#!/usr/bin/env bash
# Run the Bonsai analysis on locally-staged model files. Use this if you
# already have the GGUFs / safetensors on your desktop and don't want to
# burn the GH Actions runner.
#
# Usage:
#   scripts/run_local_analysis.sh \
#       --q1     /path/to/Bonsai-1.7B-Q1_0.gguf      \
#       --unpacked /path/to/Bonsai-1.7B-unpacked     \
#       --base   /path/to/Qwen3-1.7B                 \
#       [--family bonsai|ternary]                    \
#       [--size 1.7B|4B|8B]                          \
#       [--filter SUBSTR]
#
# --unpacked / --base may point at either:
#   - a directory containing a .safetensors / .gguf file, or
#   - the file itself.
#
# Reports are written to reports/<family>-<size>/ so you can `git add` them
# and `git push` for the same effect as the workflow's commit-back step.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAMILY=bonsai
SIZE=1.7B
FILTER=""
Q1=""
UNPACKED=""
BASE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --q1)        Q1="$2"; shift 2 ;;
    --unpacked)  UNPACKED="$2"; shift 2 ;;
    --base)      BASE="$2"; shift 2 ;;
    --family)    FAMILY="$2"; shift 2 ;;
    --size)      SIZE="$2"; shift 2 ;;
    --filter)    FILTER="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed -e 's/^# \?//' -e '$d'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$Q1" ]; then
  echo "ERROR: --q1 is required (path to the Q1_0 .gguf)." >&2
  exit 2
fi

# Resolve a possibly-directory argument to a single file with a glob.
pick_file() {
  local root="$1"; shift
  if [ -z "$root" ]; then echo ""; return 0; fi
  if [ -f "$root" ]; then echo "$root"; return 0; fi
  for ext in "$@"; do
    f=$(find "$root" -maxdepth 2 -type f -name "$ext" 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "$f"; return 0; }
  done
  echo ""
}

UNPACKED_FILE=$(pick_file "$UNPACKED" "*.safetensors" "*.gguf" || true)
BASE_FILE=$(pick_file "$BASE" "*.safetensors" "*.gguf" || true)

DEST="reports/${FAMILY}-${SIZE}"
mkdir -p "$DEST"

echo "[i] Q1_0 GGUF: $Q1"
echo "[i] Unpacked: ${UNPACKED_FILE:-<none>}"
echo "[i] Base:     ${BASE_FILE:-<none>}"
echo "[i] Output:   $DEST"

filter_arg=()
[ -n "$FILTER" ] && filter_arg=(--filter "$FILTER")

# Ensure deps. Use the project's uv environment if present, else system python.
RUN=(uv run python)
if ! command -v uv >/dev/null 2>&1; then
  RUN=(python3)
fi

echo
echo "== 01: GGUF metadata + tensor inventory =="
"${RUN[@]}" src/gguf_inspect.py "$Q1" --tensors | tee "$DEST/01_metadata.txt"

echo
echo "== 02: Q1_0 sign-pattern / sortedness analysis =="
"${RUN[@]}" src/analyze_q1_0.py "$Q1" --top 0 "${filter_arg[@]}" | tee "$DEST/02_q1_0_analysis.txt"

if [ -n "$UNPACKED_FILE" ] && [ -n "$BASE_FILE" ]; then
  echo
  echo "== 03: Bonsai-unpacked vs Qwen3 base (FP) =="
  "${RUN[@]}" src/compare_unpacked_vs_qwen3.py \
      "$UNPACKED_FILE" "$BASE_FILE" "${filter_arg[@]}" \
      | tee "$DEST/03_unpacked_vs_qwen3.txt"
else
  echo "[i] Skipping 03: need both --unpacked and --base." \
      | tee "$DEST/03_unpacked_vs_qwen3.txt"
fi

if [ -n "$UNPACKED_FILE" ] && [[ "$UNPACKED_FILE" == *.gguf ]]; then
  echo
  echo "== 04: 3-way Q1 / unpacked-GGUF / (base-GGUF if any) =="
  qw_arg=()
  if [ -n "$BASE_FILE" ] && [[ "$BASE_FILE" == *.gguf ]]; then
    qw_arg=(--qwen "$BASE_FILE")
  fi
  "${RUN[@]}" src/compare_unpacked.py \
      --bonsai-q1 "$Q1" --bonsai-fp "$UNPACKED_FILE" \
      "${qw_arg[@]}" --limit 12 \
      | tee "$DEST/04_three_way_gguf.txt"
else
  echo "[i] Skipping 04: needs an unpacked .gguf." | tee "$DEST/04_three_way_gguf.txt"
fi

# Top-line summary
{
  echo "family: $FAMILY"
  echo "size: $SIZE"
  echo "q1: $Q1"
  echo "unpacked: ${UNPACKED_FILE:-<none>}"
  echo "base: ${BASE_FILE:-<none>}"
  echo "ran_at: $(date -u +%FT%TZ)"
  echo "host: $(uname -a)"
  echo
  echo "## Q1 sha256"
  shasum -a 256 "$Q1" 2>/dev/null || sha256sum "$Q1"
} > "$DEST/run.yaml"

echo
echo "[ok] Reports written to $DEST"
echo "     git add $DEST && git commit -m 'reports: $FAMILY $SIZE (local)'"
