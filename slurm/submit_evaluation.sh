#!/bin/bash
#SBATCH --job-name=SelectiveMagnoViT-Eval
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
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

# Create logs directory
mkdir -p logs/slurm

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT environment variable not set"
    echo "Usage: CHECKPOINT=path/to/checkpoint.pth sbatch slurm/submit_evaluation.sh"
    exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"

# Run evaluation
python scripts/evaluate.py \
    --config configs/base_config.yml \
    --checkpoint "$CHECKPOINT" \
    --magno_dir "${MAGNO_DIR:-data/preprocessed/magno_images}" \
    --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
    --split "${SPLIT:-val}" \
    --output_dir "${OUTPUT_DIR:-results}" \
    --analyze_patches \
    ${EXTRA_ARGS}

echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
