#!/bin/bash
#SBATCH --job-name=SelectiveMagnoViT-Preprocess
#SBATCH --output=logs/slurm/preprocess_%j.out
#SBATCH --error=logs/slurm/preprocess_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

eval "$(mamba shell hook --shell bash)"
mamba activate drawings
echo "Environment activated successfully"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs/slurm

# Run preprocessing
python scripts/preprocess.py \
    --config configs/base_config.yml \
    --raw_data_root "${RAW_DATA_ROOT:-data/raw_dataset}" \
    --preprocessed_root "${PREPROCESSED_ROOT:-data/preprocessed}" \
    --splits train val \
    ${EXTRA_ARGS}

echo "=========================================="
echo "Preprocessing completed at: $(date)"
echo "=========================================="
