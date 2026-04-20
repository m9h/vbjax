#!/bin/bash
#SBATCH --job-name=vbjax-analytical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/analytical/slurm-%j.out
#SBATCH --error=results/analytical/slurm-%j.err

set -euo pipefail

export TMPDIR=/data/mhough/tmp
export XLA_FLAGS="--xla_gpu_deterministic_ops=false"

cd ~/dev/vbjax
source .venv/bin/activate

mkdir -p results/analytical

echo "=== Analytical Transfer Function BMS ==="
echo "Job ID: $SLURM_JOB_ID"
date

python -c "import jax; print('Device:', jax.devices()[0])"

python scripts/dgx_bms_analytical.py --alpha-blocking --bic --output results/analytical

echo "=== Done ==="
date
