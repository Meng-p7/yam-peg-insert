#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Checking OS and Python isolation..."
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "Detected OS: ${PRETTY_NAME:-unknown}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[2/4] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[2/4] uv already installed: $(uv --version)"
fi

echo "[3/4] Creating isolated Python 3.12 virtualenv (.venv) without touching ROS 2 Python 3.10..."
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "[4/4] Running pytest and CPU smoke test..."
pytest
python scripts/smoke_test.py

echo ""
echo "Setup complete! Activate anytime with: source .venv/bin/activate"
echo "If GUI viewer fails without a discrete GPU, run: export LIBGL_ALWAYS_SOFTWARE=1"
