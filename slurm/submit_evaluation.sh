#!/bin/bash
#SBATCH --job-name=SelectiveMagnoViT-Eval
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

# Exit on error

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

# Setup environment
eval "$(mamba shell hook --shell bash)"
mamba activate vla 
echo "Environment activated successfully"

# Verify GPU
nvidia-smi

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT environment variable not set"
    echo "Usage: CHECKPOINT=path/to/checkpoint.pth sbatch slurm/submit_evaluation.sh"
    exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"

# Create unique output directory for this evaluation run
CHECKPOINT_NAME=$(basename "$CHECKPOINT" .pth)
SPLIT_NAME="${SPLIT:-val}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/eval_${CHECKPOINT_NAME}_${SPLIT_NAME}_${TIMESTAMP}"

echo "Results will be saved to: $RUN_DIR"
mkdir -p "$RUN_DIR"

# Run evaluation
echo "Running evaluation..."
python scripts/evaluate.py \
    --config configs/base_config.yml \
    --checkpoint "$CHECKPOINT" \
    --color_dir "${COLOR_DIR:-data/preprocessed/color_images}" \
    --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
    --split "$SPLIT_NAME" \
    --output_dir "$RUN_DIR" \
    --analyze_patches \
    ${EXTRA_ARGS}

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

# Find the results JSON file
RESULTS_FILE="$RUN_DIR/evaluation_${CHECKPOINT_NAME}_${SPLIT_NAME}.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "WARNING: Results file not found at $RESULTS_FILE"
    echo "Skipping visualization step"
else
    echo "=========================================="
    echo "Generating visualizations..."
    echo "=========================================="

    python scripts/visualize_results.py \
        --config configs/base_config.yml \
        --checkpoint "$CHECKPOINT" \
        --results_file "$RESULTS_FILE" \
        --color_dir "${COLOR_DIR:-data/preprocessed/color_images}" \
        --lines_dir "${LINES_DIR:-data/preprocessed/line_drawings}" \
        --split "$SPLIT_NAME" \
        --output_dir "$RUN_DIR/visualizations" \
        --all \
        --num_samples "${NUM_VIZ_SAMPLES:-10}"

    if [ $? -eq 0 ]; then
        echo "Visualizations saved to: $RUN_DIR/visualizations"
    else
        echo "WARNING: Visualization step failed"
    fi
fi

echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "All results saved to: $RUN_DIR"
echo "=========================================="
