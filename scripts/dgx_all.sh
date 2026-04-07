#!/bin/bash
# One-shot: clone, setup, and run vbjax BMS on DGX Spark
set -euo pipefail

REPO=https://github.com/m9h/vbjax.git
BRANCH=feature/cmc-linearization
WORKDIR=$HOME/dev/vbjax

echo "=== vbjax BMS on DGX Spark ==="
date

# Clone or update
if [ -d "$WORKDIR" ]; then
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    mkdir -p "$(dirname "$WORKDIR")"
    git clone -b "$BRANCH" "$REPO" "$WORKDIR"
    cd "$WORKDIR"
fi

# Setup environment
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate

uv pip install "jax[cuda12]" -q
uv pip install -e ".[dev]" -q

# Verify
python -c "import jax; print('Device:', jax.devices()[0]); import vbjax; print('vbjax OK')"

# Run tests
echo ""
echo "=== Tests ==="
python -m pytest vbjax/tests/ -x -q --tb=short -k "not slow" 2>&1 | tail -3

# Quick BMS
echo ""
echo "=== Quick BMS ==="
python scripts/dgx_bms.py --quick --output results/quick

# Full BMS
echo ""
echo "=== Full BMS ==="
python scripts/dgx_bms.py --output results/baseline

# Full BMS + pharmacological perturbation
echo ""
echo "=== Full BMS + Pharma ==="
python scripts/dgx_bms.py --pharma --output results/pharma

echo ""
echo "=== Done ==="
date
ls -lh results/*.json
