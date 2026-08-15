#!/usr/bin/env bash
# Bootstrap the keqing-mortal dev environment on Linux/macOS.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${1:-.venv}"
PY="$REPO/$VENV/bin/python"

# 1. Project venv + dependencies. `uv sync` also installs the dev group and
#    the editable project itself; every later command should use `uv run`.
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV"
fi
uv sync --python "$PY"

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

# 2. libriichi runtime, built from the vendored Mortal crate.
#    The default feature set also links mimalloc; with current rust-lld that
#    produces non-PIC static objects in a cdylib, so build the pymod feature
#    explicitly for this Linux extension.
NATIVE_ROOT="$REPO/third_party/Mortal"
D3_PATCH="$REPO/training/mortal/patches/libriichi_d3_decision_context.patch"
CONTEXT_SOURCE="$NATIVE_ROOT/libriichi/src/agent/defs.rs"
if [ -f "$CONTEXT_SOURCE" ] && [ -f "$D3_PATCH" ]; then
  if ! grep -q "pub struct DecisionContext" "$CONTEXT_SOURCE"; then
    echo "Applying D3 decision-context patch to the local Mortal checkout..."
    git -C "$NATIVE_ROOT" apply --whitespace=nowarn -- "$D3_PATCH"
  fi
fi
TARGET_SO="$NATIVE_ROOT/target/release/libriichi.so"
if [ ! -f "$TARGET_SO" ]; then
  cargo build --manifest-path "$NATIVE_ROOT/Cargo.toml" -p libriichi --lib --release \
    --no-default-features --features pymod
fi
cp "$TARGET_SO" "$SITE/riichi.so"
rm -rf "$SITE/libriichi"
cp -R "$REPO/third_party/libriichi/libriichi" "$SITE/libriichi"
"$PY" -c "from libriichi.arena import OneVsThree; assert hasattr(OneVsThree, 'py_selfplay'); print('libriichi OK')"

# 3. keqing_core wheel: build and install. Publishing the shared copy to
#    KEQING_DATA_ROOT is intentionally left to the Workbench-side step.
WHEEL="$REPO/rust/keqing_core/target/wheels/keqing_core-0.1.0-cp*.whl"
if ! ls $WHEEL >/dev/null 2>&1; then
  "$PY" "$REPO/rust/keqing_core/build.py"
fi
uv pip install --python "$PY" $WHEEL
"$PY" -c "import keqing_core; assert keqing_core.is_available(); print('keqing_core OK (rust available)')"

echo "setup complete: $VENV (use 'uv run ...' from $REPO)"
