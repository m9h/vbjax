#!/bin/bash
#SBATCH --job-name=vbjax-realdata-bms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/realdata/slurm-%j.out
#SBATCH --error=results/realdata/slurm-%j.err

set -euo pipefail

cd ~/dev/vbjax
source .venv/bin/activate

mkdir -p results/realdata

echo "=== vbjax Real EEG BMS (Hartoyo et al.) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

python -c "import jax; print('Device:', jax.devices()[0])"

# Full 82 subjects, eyes-closed + alpha-blocking
python scripts/dgx_bms_realdata.py --alpha-blocking --output results/realdata

echo "=== Done ==="
date
