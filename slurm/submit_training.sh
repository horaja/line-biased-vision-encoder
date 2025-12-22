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

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

module list

# Setup environment
eval "$(mamba shell hook --shell bash)"
mamba activate vla 
echo "Environment activated successfully"


# Change to project directory
cd $SLURM_SUBMIT_DIR

PP=1.0

# Run training with unbuffered output
python -u scripts/train.py \
    --config configs/base_config.yml \
    --color_dir "${COLOR_DIR:-data/preprocessed/color_images}" \
    --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
    --output_dir "checkpoints/p${PP}" \
    --patch_percentage "$PP" \
    --epochs 100

mamba deactivate

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="