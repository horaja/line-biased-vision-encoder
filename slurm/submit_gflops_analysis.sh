#!/bin/bash
#SBATCH --job-name=gflops_analysis
#SBATCH --output=logs/slurm/gflops_%j.log
#SBATCH --error=logs/slurm/gflops_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# 1. Setup Environment
eval "$(mamba shell hook --shell bash)"
mamba activate vla

# 2. Run Analysis
echo "Starting GFLOPS analysis on $(hostname) at $(date)"
python scripts/measure_gflops_scaling.py
echo "Finished at $(date)"