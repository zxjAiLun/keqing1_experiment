#!/usr/bin/env bash
# Bootstrap the keqing-mortal dev environment on Linux/macOS.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${1:-.venv}"
PY="$REPO/$VENV/bin/python"

if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV"
fi

uv pip install --python "$PY" "numpy>=1.24" "riichienv==0.4.8" "torch>=2.11.0" \
  "tensorboard>=2.20.0" "pytest>=9.0.2" "ruff>=0.15.10"

WHEEL="$REPO/rust/keqing_core/target/wheels/keqing_core-0.1.0-cp*.whl"
if ! ls $WHEEL >/dev/null 2>&1; then
  "$PY" "$REPO/rust/keqing_core/build.py"
fi
uv pip install --python "$PY" $WHEEL

echo "setup complete: $VENV"
