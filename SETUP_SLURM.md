# SLURM Cluster Setup Guide

Complete step-by-step guide for setting up and running SelectiveMagnoViT on a SLURM cluster.

---

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Conda or Mamba package manager (Mamba recommended for faster installs)
- Internet access from login node (for downloads)

---

## Step 1: Environment Setup (Login Node)

### 1.1 Navigate to project directory
```bash
cd /user_data/horaja/workspace/magno_stream_encoder
```

### 1.2 (Optional) Install Mamba for faster package management
```bash
# If mamba is not already available
conda install -n base -c conda-forge mamba
```

Mamba is a faster drop-in replacement for conda. If you prefer to use conda, just replace `mamba` with `conda` in all commands below.

### 1.3 Create environment
```bash
mamba env create -f environment.yml
```

This will create an environment named `drawings` with all dependencies.

**Using conda instead:** If you don't want to use mamba, simply run:
```bash
conda env create -f environment.yml
```

### 1.4 Activate environment and install package
```bash
mamba activate drawings  # or: conda activate drawings
pip install -e .
```

### 1.5 Quick verification (login node)
```bash
# Test that basic imports work
python -c "import selective_magno_vit; print('✓ Package installed successfully')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
```

---

## Step 2: Download Dataset (Login Node)

### Option A: Use the helper script (Recommended)
```bash
bash scripts/download_dataset.sh
```

This interactive script will:
- Download ImageNette dataset (~1.4GB)
- Download informative-drawings pretrained model (~100MB)
- Verify directory structure

### Option B: Manual download

**Download ImageNette:**
```bash
mkdir -p data/raw_dataset
cd data/raw_dataset
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
mv imagenette2-320/* .
rmdir imagenette2-320
cd ../..
```

**Download pretrained model:**
```bash
cd third_party/informative-drawings/checkpoints
pip install gdown  # If not already installed
gdown 1MIdHzecxz-z0uY3ARL_R40DlKcuQxiDk -O model.zip
unzip model.zip
cd ../../..
```

### Verify downloads
```bash
# Check dataset
ls data/raw_dataset/train/  # Should show 10 class directories
ls data/raw_dataset/val/    # Should show 10 class directories

# Check model
ls third_party/informative-drawings/checkpoints/opensketch_style/
```

---

## Step 3: Run Tests (SLURM Compute Node)

### 3.1 Create logs directory
```bash
mkdir -p logs/slurm
```

### 3.2 Submit test job
```bash
sbatch slurm/submit_tests.sh
```

### 3.3 Monitor job
```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with actual job ID)
tail -f logs/slurm/test_JOBID.out

# Or view after completion
cat logs/slurm/test_JOBID.out
```

**Expected output:**
- Package imports successful
- All unit tests pass (or warnings about GPU tests being skipped)

---

## Step 4: Run Preprocessing (SLURM Compute Node)

This generates line drawings and magno images from your raw dataset.

### 4.1 Check/update configuration
Verify paths in `configs/base_config.yml`:
```yaml
data:
  raw_root: "/user_data/horaja/workspace/magno_stream_encoder/data/raw_dataset"
  preprocessed_root: "/user_data/horaja/workspace/magno_stream_encoder/data/preprocessed"
```

### 4.2 Submit preprocessing job
```bash
sbatch slurm/submit_preprocessing.sh
```

**Note:** This job requests:
- 1 GPU
- 32GB RAM
- 12 hours
- Adjust in `slurm/submit_preprocessing.sh` if needed

### 4.3 Monitor preprocessing
```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f logs/slurm/preprocess_JOBID.out

# Check for errors
tail -f logs/slurm/preprocess_JOBID.err
```

**Expected duration:** 2-4 hours for ImageNette

### 4.4 Verify preprocessing output
```bash
# Check that output directories were created
ls data/preprocessed/

# Should see:
# - magno_images/
# - line_drawings/
# - color_images/

# Check contents
ls data/preprocessed/magno_images/train/
ls data/preprocessed/line_drawings/train/
```

---

## Step 5: Run Quick Training Test (SLURM Compute Node)

Before running full training, test that everything works with a short 2-epoch run.

### 5.1 Submit quick test
```bash
sbatch slurm/submit_quick_train_test.sh
```

This runs training for just 2 epochs to verify:
- Data loads correctly
- Model trains without errors
- Checkpointing works

### 5.2 Monitor
```bash
squeue -u $USER
tail -f logs/slurm/quick_train_JOBID.out
```

**Expected duration:** ~10 minutes

### 5.3 Check outputs
```bash
ls checkpoints/quick_test/  # Should contain checkpoint files
```

---

## Step 6: Run Full Training (SLURM Compute Node)

Once quick test passes, run full training.

### 6.1 (Optional) Customize training config
Edit `configs/training_config.yml` to adjust:
- Number of epochs
- Batch size
- Learning rate
- Patch percentage
- etc.

### 6.2 Submit training job
```bash
sbatch slurm/submit_training.sh
```

**Default resources:**
- 1 GPU
- 32GB RAM
- 8 hours
- Adjust in `slurm/submit_training.sh` if needed

### 6.3 Monitor training
```bash
# Check job status
squeue -u $USER

# Watch training progress
tail -f logs/slurm/train_JOBID.out

# Monitor TensorBoard (requires port forwarding)
# On login node:
tensorboard --logdir logs/tensorboard --port 6006

# Then on your local machine:
ssh -L 6006:localhost:6006 horaja@your-cluster.edu
# Open http://localhost:6006 in browser
```

---

## Step 7: Evaluate Model (SLURM Compute Node)

### 7.1 Find best checkpoint
```bash
ls -lht checkpoints/  # Look for best_model.pth or similar
```

### 7.2 Submit evaluation job
```bash
export CHECKPOINT="checkpoints/best_model.pth"
sbatch slurm/submit_evaluation.sh
```

### 7.3 View results
```bash
# Results saved to results/
cat results/evaluation_best_model_val.json
```

---

## Common SLURM Commands

### Job Management
```bash
# Submit job
sbatch script.sh

# Check your jobs
squeue -u $USER

# Cancel job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# View job details
scontrol show job JOBID

# View completed job info
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

### Cluster Information
```bash
# View partition info
sinfo

# View node details
sinfo -N -l

# Check partition features
sinfo -o "%20P %5a %10l %6D %6t %8N %f"

# Check available resources
squeue -p gpu --Format=Reason,NumNodes,NumCPUs,tres-per-node
```

---

## Troubleshooting

### Issue: Job stuck in pending
**Check why:**
```bash
squeue -u $USER
scontrol show job JOBID | grep Reason
```

**Common reasons:**
- `Resources`: No nodes available - wait or reduce resource request
- `Priority`: Other jobs have higher priority - wait
- `QOSMaxJobsPerUserLimit`: Too many jobs running - cancel some

**Solution:** Adjust resource requests in SLURM script or wait

---

### Issue: Out of memory
**Symptoms:** Job fails with "Out of Memory" or killed status

**Solutions:**
1. Increase `--mem` in SLURM script
2. Reduce batch size in config
3. Use smaller model

---

### Issue: Mamba/Conda environment not activating
**Solutions:**
```bash
# Initialize conda for bash
conda init bash
source ~/.bashrc

# Or check if cluster uses modules
module load conda

# If mamba is not found, install it
conda install -n base -c conda-forge mamba
```

---

### Issue: GPU not found
**Check GPU availability:**
```bash
sinfo -o "%20P %5a %10l %6D %6t %8N %f" | grep gpu
```

**Solutions:**
1. Verify partition name (might be `gpu`, `gpu-v100`, etc.)
2. Update `--partition=` in SLURM scripts
3. Check if your account has GPU access

---

### Issue: File not found errors
**Solutions:**
1. Use absolute paths in configs
2. Verify `cd $SLURM_SUBMIT_DIR` in scripts
3. Check file permissions

---

## Quick Reference

### File Structure
```
magno_stream_encoder/
├── data/
│   ├── raw_dataset/          # Downloaded ImageNette
│   └── preprocessed/         # Generated by preprocessing
├── slurm/
│   ├── submit_tests.sh       # Test installation
│   ├── submit_preprocessing.sh  # Generate line drawings
│   ├── submit_quick_train_test.sh  # Quick training test
│   ├── submit_training.sh    # Full training
│   └── submit_evaluation.sh  # Model evaluation
├── scripts/
│   ├── download_dataset.sh   # Helper to download data
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── visualize_results.py # Visualization
└── logs/
    └── slurm/               # SLURM job outputs
```

### Typical Workflow
```bash
# 1. Setup (once)
mamba env create -f environment.yml  # or: conda env create -f environment.yml
mamba activate drawings              # or: conda activate drawings
pip install -e .

# 2. Download data (once)
bash scripts/download_dataset.sh

# 3. Test
sbatch slurm/submit_tests.sh

# 4. Preprocess (once per dataset)
sbatch slurm/submit_preprocessing.sh

# 5. Quick test
sbatch slurm/submit_quick_train_test.sh

# 6. Train
sbatch slurm/submit_training.sh

# 7. Evaluate
export CHECKPOINT="checkpoints/best_model.pth"
sbatch slurm/submit_evaluation.sh
```

---

## Getting Help

- **SLURM documentation:** https://slurm.schedmd.com/
- **Your cluster docs:** Check your institution's HPC documentation
- **Support:** Contact your cluster's help desk for cluster-specific issues

---

## Next Steps

After successful training:
1. Run evaluation on test set
2. Generate visualizations with `scripts/visualize_results.py`
3. Experiment with different hyperparameters
4. Try different patch selection strategies

Good luck with your experiments! 🚀
