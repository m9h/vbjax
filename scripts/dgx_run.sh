#!/bin/bash
# Run the full Bayesian model comparison on DGX Spark
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Run tests first to verify everything works
echo "=== Running tests ==="
python -m pytest vbjax/tests/ -x -q --tb=short -k "not slow" 2>&1 | tail -5

# Quick sanity check (should complete in ~5 min)
echo ""
echo "=== Quick BMS sanity check ==="
python scripts/dgx_bms.py --quick --output results/quick

# Full comparison — baseline only (~30 min estimated)
echo ""
echo "=== Full BMS comparison ==="
python scripts/dgx_bms.py --output results/baseline

# Full comparison with pharmacological perturbation (~45 min estimated)
echo ""
echo "=== Full BMS with pharma perturbation ==="
python scripts/dgx_bms.py --pharma --output results/pharma

echo ""
echo "=== All runs complete ==="
echo "Results in: results/"
ls -lh results/*.json
