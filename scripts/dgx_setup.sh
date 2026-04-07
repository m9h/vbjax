#!/bin/bash
# DGX Spark setup for vbjax Bayesian model comparison
# Run once after cloning the repo on the DGX Spark
set -euo pipefail

echo "=== vbjax DGX Spark setup ==="

cd "$(dirname "$0")/.."

# Create venv with uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12
source .venv/bin/activate

# Install JAX with CUDA support (GB10 = CUDA 12.x)
uv pip install "jax[cuda12]"

# Install vbjax in dev mode
uv pip install -e ".[dev]"

# Install optax for advanced optimization
uv pip install optax

# Verify
python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'Platform: {jax.local_devices()[0].platform}')
import vbjax as vb
print(f'vbjax {vb.__version__} loaded')
from vbjax.bms import bms_ffx, bms_rfx
from vbjax.spectral import welch_psd_jax
print('All modules OK')
"

echo "=== Setup complete ==="
