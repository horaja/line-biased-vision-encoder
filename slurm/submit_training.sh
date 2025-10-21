#!/bin/bash
#SBATCH --job-name=SelectiveMagnoViT
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

# Exit on error
set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

# Setup environment
eval "$(mamba shell hook --shell bash)"
mamba env update -f environment.yml 2>/dev/null || mamba env create -f environment.yml -y
mamba activate drawings

# Verify GPU
nvidia-smi

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run training
python scripts/train.py \
    --config configs/base_config.yaml \
    --magno_dir "${MAGNO_DIR:-data/preprocessed/magno_images}" \
    --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
    --output_dir "${OUTPUT_DIR:-checkpoints}" \
    ${EXTRA_ARGS}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="