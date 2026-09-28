#!/usr/bin/env bash
# Usage: scripts/new-comp.sh <competition-slug> [--no-download]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG="${1:?competition slug required}"
DEST="$ROOT/competitions/$SLUG"

[[ -e "$DEST" ]] && { echo "already exists: $DEST" >&2; exit 1; }
cp -R "$ROOT/templates/competition" "$DEST"
find "$DEST" -type f \( -name '*.md' -o -name '*.toml' \) -exec sed -i '' "s/__SLUG__/$SLUG/g" {} +
mkdir -p "$DEST/data" "$DEST/outputs"

(cd "$DEST" && uv sync -q)

if [[ "${2:-}" != "--no-download" ]]; then
  kaggle competitions download -c "$SLUG" -p "$DEST/data" && \
    (cd "$DEST/data" && for z in *.zip; do [[ -e "$z" ]] && unzip -q -o "$z" && rm "$z"; done)
fi
echo "created $DEST"
