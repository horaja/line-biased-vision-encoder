#!/bin/bash
# SLURM script to run lightweight pytest suite (runtime line drawing helpers).

#SBATCH --job-name=SelectiveMagnoViT-Pytest
#SBATCH --output=logs/slurm/pytest_%j.out
#SBATCH --error=logs/slurm/pytest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=00:20:00
#SBATCH --partition=cpu

set -euo pipefail

echo "=========================================="
echo "SLURM Pytest Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p logs/slurm

# Activate environment
eval "$(mamba shell hook --shell bash)"
# mamba env update -f environment.yml 2>/dev/null || mamba env create -f environment.yml -y
mamba activate vla

# Move to repo root
cd "$SLURM_SUBMIT_DIR"

echo "Running pytest (runtime line drawing suite)..."
pytest tests/test_line_drawings.py tests/test_dataset_runtime.py -v --maxfail=1 --disable-warnings

echo "=========================================="
echo "Pytest completed at $(date)"
echo "=========================================="
