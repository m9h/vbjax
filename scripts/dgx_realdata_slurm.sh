#!/bin/bash
#SBATCH --job-name=vbjax-realdata-bic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/realdata/slurm-%j.out
#SBATCH --error=results/realdata/slurm-%j.err

set -euo pipefail

export TMPDIR=/data/mhough/tmp

cd ~/dev/vbjax
source .venv/bin/activate

mkdir -p results/realdata

echo "=== vbjax Real EEG BMS — BIC (Hartoyo et al.) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

python -c "import jax; print('Device:', jax.devices()[0])"

# Full 82 subjects, eyes-closed + alpha-blocking, BIC approximation
python scripts/dgx_bms_realdata.py --bic --alpha-blocking --output results/realdata

echo "=== Done ==="
date
