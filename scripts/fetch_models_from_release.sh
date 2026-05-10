#!/usr/bin/env bash
# Fetch the model artifacts a previous workflow run uploaded to a GitHub
# Release back into the local working tree under models/<family>/<size>/.
#
# This is the symmetric counterpart of the workflow's "Upload models to a
# GitHub Release" step: it walks the release's assets, downloads any chunked
# files (*.part-*) and reassembles them, and verifies sha256 against
# SHA256SUMS.staged.txt if present.
#
# Future Claude sessions in the web sandbox use this because:
#   * github.com + release-assets.githubusercontent.com are allowlisted
#     (verified empirically)
#   * huggingface.co is NOT allowlisted, so we can't fetch from prism-ml/*
#
# Usage:
#   scripts/fetch_models_from_release.sh             # pick latest models-* tag
#   scripts/fetch_models_from_release.sh <tag>       # specific release tag
#   FAMILY=bonsai SIZE=1.7B scripts/fetch_models_from_release.sh
#       (auto-resolves to the most recent matching tag)

set -euo pipefail

REPO=${REPO:-tg-techie/reversing-bonsai}
DEST_ROOT=${DEST_ROOT:-models}

API="https://api.github.com/repos/$REPO"

resolve_tag() {
  local prefix="$1"
  # Returns most recent release tag matching prefix
  curl -sL -m 10 "$API/releases?per_page=30" \
    | python3 -c "
import json, sys
prefix=$(printf '%s' "$prefix" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
data=json.load(sys.stdin)
for r in data:
    t=r.get('tag_name','')
    if t.startswith(prefix):
        print(t); sys.exit(0)
"
}

TAG="${1:-}"
if [ -z "$TAG" ]; then
  fam="${FAMILY:-bonsai}"
  sz="${SIZE:-1.7B}"
  PREFIX="models-${fam}-${sz}-r"
  TAG=$(resolve_tag "$PREFIX")
  if [ -z "$TAG" ]; then
    echo "ERROR: no release with tag prefix '$PREFIX' found in $REPO" >&2
    exit 2
  fi
  echo "[i] auto-resolved tag: $TAG"
fi

# Parse family + size from tag (models-<family>-<size>-r<n>)
FAMILY=$(echo "$TAG" | sed -E 's/^models-([a-zA-Z]+)-.*/\1/')
SIZE=$(echo "$TAG"   | sed -E 's/^models-[a-zA-Z]+-([0-9.]+B)-r.*/\1/')

DEST="$DEST_ROOT/$FAMILY/$SIZE"
mkdir -p "$DEST"
cd "$DEST"

echo "[i] release tag: $TAG  -> destination: $(pwd)"

# Get list of asset URLs
mapfile -t ASSETS < <(curl -sL -m 15 "$API/releases/tags/$TAG" \
  | python3 -c "
import json, sys
d=json.load(sys.stdin)
for a in d.get('assets',[]):
    print(a['browser_download_url'])
")
if [ "${#ASSETS[@]}" -eq 0 ]; then
  echo "ERROR: no assets attached to $TAG" >&2
  exit 1
fi

# Download each asset
for url in "${ASSETS[@]}"; do
  name=$(basename "$url")
  if [ -f "$name" ]; then
    echo "[skip] $name already present"
  else
    echo "[get] $name"
    curl -sLf -m 600 -o "$name.tmp" "$url"
    mv "$name.tmp" "$name"
  fi
done

# Reassemble any chunked files. Convention: <orig>.part-NNN where N is digits.
shopt -s nullglob
for first in *.part-000; do
  base="${first%.part-000}"
  if [ -f "$base" ]; then
    echo "[skip] $base already reassembled"
  else
    echo "[join] $base from chunks"
    cat "$base".part-* > "$base.tmp"
    mv "$base.tmp" "$base"
  fi
done

# Verify sha256 if SHA256SUMS.staged.txt is present.
if [ -f SHA256SUMS.staged.txt ]; then
  echo "[verify] SHA256SUMS.staged.txt"
  # Only verify entries that match files actually on disk (chunks may
  # have been removed after reassembly, that's fine).
  awk '{
    file=$2; sub(/^[*]/, "", file);
    cmd="test -f " file
    if (system(cmd)==0) print
  }' SHA256SUMS.staged.txt > .sums.present
  if [ -s .sums.present ]; then
    sha256sum --quiet -c .sums.present
    echo "[ok] sha256 verified for $(wc -l < .sums.present) files"
  fi
  rm -f .sums.present
fi

echo
echo "[done] models fetched to: $(pwd)"
ls -lh
