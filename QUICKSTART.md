# Quick Start Guide - SLURM Cluster

Fast-track setup for SelectiveMagnoViT on SLURM clusters.

## Prerequisites
- SLURM cluster with GPU nodes
- Conda available

---

## 🚀 5-Minute Setup

### 1. Environment (Login Node)
```bash
cd /user_data/horaja/workspace/magno_stream_encoder
mamba env create -f environment.yml
mamba activate drawings
pip install -e .
```

**Note:** If mamba is not installed, install it first with `conda install -n base -c conda-forge mamba`, or use `conda env create -f environment.yml` instead.

### 2. Download Data (Login Node)
```bash
bash scripts/download_dataset.sh
# Choose option 3 (Both dataset and model)
```

### 3. Run Tests (Submit to SLURM)
```bash
mkdir -p logs/slurm
sbatch slurm/submit_tests.sh
squeue -u $USER  # Check status
```

### 4. Preprocess Data (Submit to SLURM)
```bash
sbatch slurm/submit_preprocessing.sh
# Wait ~2-4 hours for ImageNette
```

### 5. Quick Training Test (Submit to SLURM)
```bash
sbatch slurm/submit_quick_train_test.sh
# Verify everything works (~10 min)
```

### 6. Full Training (Submit to SLURM)
```bash
sbatch slurm/submit_training.sh
# Wait for training to complete
```

### 7. Evaluate
```bash
export CHECKPOINT="checkpoints/best_model.pth"
sbatch slurm/submit_evaluation.sh
```

---

## 📋 Essential Commands

### Monitor Jobs
```bash
squeue -u $USER              # Check your jobs
tail -f logs/slurm/train_*.out  # Watch training output
scancel JOBID                # Cancel a job
```

### Check Results
```bash
ls checkpoints/              # View saved models
cat results/*.json           # View evaluation results
tensorboard --logdir logs/tensorboard  # View training curves
```

---

## 🔧 Common Adjustments

### Change Training Settings
Edit `configs/training_config.yml`:
```yaml
training:
  epochs: 50                 # Fewer epochs
  batch_size: 16             # Smaller batch
  learning_rate: 1.0e-4      # Different LR
```

### Adjust SLURM Resources
Edit `slurm/submit_training.sh`:
```bash
#SBATCH --gres=gpu:2         # Use 2 GPUs
#SBATCH --mem=64gb           # More memory
#SBATCH --time=12:00:00      # More time
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Job pending | `scontrol show job JOBID` - check why |
| Out of memory | Reduce batch size or increase `--mem` |
| Can't activate mamba/conda | `conda init bash && source ~/.bashrc` |
| GPU not found | Check partition: `sinfo \| grep gpu` |

---

## 📚 Full Documentation

- **Complete setup guide:** [SETUP_SLURM.md](SETUP_SLURM.md)
- **Project README:** [README.md](README.md)
- **Configuration:** [configs/](configs/)

---

## 🎯 Expected Timeline

| Step | Time | Location |
|------|------|----------|
| Environment setup | 10-15 min | Login node |
| Download data | 5-10 min | Login node |
| Run tests | 5-10 min | SLURM |
| Preprocessing | 2-4 hours | SLURM |
| Quick train test | 10 min | SLURM |
| Full training | 2-8 hours | SLURM |

---

**Need help?** See [SETUP_SLURM.md](SETUP_SLURM.md) for detailed troubleshooting.
