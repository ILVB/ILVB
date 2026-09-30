#!/usr/bin/env sh
# Clean-install test: fresh clone → new venv → locked install → doctor → demo.
#
# Usage: scripts/clean_install_test.sh [REPO_URL_OR_PATH]
#   EXTRAS="rapid gui"   extras to install (default: rapid gui; add ocr lama … as needed)
#   KEEP=1               keep the temporary directory for inspection
# Needs git and uv (https://docs.astral.sh/uv/); falls back to python -m venv + pip.
set -eu
REPO="${1:-$(git rev-parse --show-toplevel)}"
EXTRAS="${EXTRAS:-rapid gui}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mangaar-clean-XXXXXX")"
if [ "${KEEP:-0}" != "1" ]; then trap 'rm -rf "$WORK"' EXIT; fi
echo "== clone $REPO → $WORK/src"
git clone --quiet "$REPO" "$WORK/src"
cd "$WORK/src"
export MANGAAR_CACHE_DIR="$WORK/cache"   # empty model cache: a genuine first run
if command -v uv >/dev/null 2>&1; then
  echo "== uv venv + uv sync --locked (extras: $EXTRAS)"
  uv venv --quiet --python 3.11 .venv
  set --
  for extra in $EXTRAS; do set -- "$@" --extra "$extra"; done
  uv sync --quiet --locked "$@"
else
  echo "== python -m venv + pip (uv not found)"
  python3 -m venv .venv
  spec=$(echo "$EXTRAS" | tr ' ' ',')
  .venv/bin/pip install --quiet -e ".[$spec]"
  .venv/bin/pip uninstall --quiet -y opencv-python opencv-contrib-python || true
  .venv/bin/pip install --quiet --force-reinstall opencv-python-headless
fi
echo "== manga-arabic --version"
.venv/bin/manga-arabic --version
echo "== manga-arabic doctor --no-network"
.venv/bin/manga-arabic doctor --no-network
echo "== manga-arabic demo"
.venv/bin/manga-arabic demo -o "$WORK/demo"
test -s "$WORK/demo/output/demo_zh_ar.png"
echo "CLEAN INSTALL OK ($WORK)"
