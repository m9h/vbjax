#!/bin/bash
#SBATCH --job-name=vbjax-pharma-bms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/pharma/slurm-%j.out
#SBATCH --error=results/pharma/slurm-%j.err

set -euo pipefail

cd ~/dev/vbjax
source .venv/bin/activate

mkdir -p results/pharma

echo "=== vbjax Pharma BMS via Slurm ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

python -c "import jax; print('Device:', jax.devices()[0])"

python scripts/dgx_bms.py --pharma --output results/pharma

echo "=== Done ==="
date
