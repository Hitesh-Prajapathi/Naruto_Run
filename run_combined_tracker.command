#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

export XDG_CACHE_HOME="$SCRIPT_DIR/.cache"
export MPLCONFIGDIR="$SCRIPT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/cv_model/data/venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/cv_model/data/venv/bin/python"
else
  echo "No project Python environment found."
  echo "Create one with: python3 -m venv .venv"
  echo "Then install: .venv/bin/python -m pip install -r requirements-runtime.txt"
  read -r "?Press Enter to close..."
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/cv_model/run_combined_camera.py" "$@"
