#!/bin/bash
# Digitize a long list of pages in chunks with a pause between them, so a laptop is not loaded back to back for hours.
# Resumable: src.digitize.run skips every page that already has a page.json.
# Usage: scripts/run_in_chunks.sh <list file> <pages per chunk> <pause seconds> <out dir> <layout>
set -u
LIST=$1; N=$2; PAUSE=$3; OUT=$4; LAYOUT=$5
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
split -l "$N" -d -a 3 "$LIST" "$TMP/chunk."
TOTAL=$(ls "$TMP" | wc -l)
i=0
for f in "$TMP"/chunk.*; do
  i=$((i + 1))
  echo "$(date '+%F %T') chunk $i/$TOTAL start"
  uv run python -m src.digitize.run --config configs/digitize.yml --layout "$LAYOUT" --out-dir "$OUT" --list "$f"
  echo "$(date '+%F %T') chunk $i/$TOTAL done"
  [ "$i" -lt "$TOTAL" ] && sleep "$PAUSE"
done
rm -rf "$TMP"
echo "$(date '+%F %T') all chunks done"
