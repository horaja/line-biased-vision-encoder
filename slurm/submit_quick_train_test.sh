#!/bin/bash
#SBATCH --job-name=QuickTrainTest
#SBATCH --output=logs/slurm/quick_train_%j.out
#SBATCH --error=logs/slurm/quick_train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Exit on error
set -e

echo "=========================================="
echo "Quick Training Test"
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
echo ""
echo "GPU Information:"
nvidia-smi

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p checkpoints/quick_test
mkdir -p logs/quick_test

echo ""
echo "Starting quick training test (2 epochs)..."
echo "This is just to verify training pipeline works"
echo "----------------------------------------"

# Run training for just 2 epochs as a test
python scripts/train.py \
    --config configs/base_config.yml \
    --magno_dir "${MAGNO_DIR:-data/preprocessed/magno_images}" \
    --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
    --output_dir checkpoints/quick_test \
    --epochs 2 \
    --batch_size 16 \
    --patch_percentage 0.4 \
    --patience 10

echo ""
echo "=========================================="
echo "Quick training test completed!"
echo "Check checkpoints/quick_test/ for outputs"
echo "Completion time: $(date)"
echo "=========================================="
