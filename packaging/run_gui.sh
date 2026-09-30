#!/usr/bin/env sh
# MangaAR GUI launcher (macOS/Linux). Uses the project venv if present, else PATH.
# Opens http://127.0.0.1:7860 (local only). Extra arguments go to `manga-arabic gui`.
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(dirname -- "$HERE")"
if [ -x "$ROOT/.venv/bin/manga-arabic" ]; then
  exec "$ROOT/.venv/bin/manga-arabic" gui "$@"
elif command -v manga-arabic >/dev/null 2>&1; then
  exec manga-arabic gui "$@"
else
  echo "manga-arabic is not installed. See README.md (Install)." >&2
  exit 1
fi
