#!/bin/bash
# Dataset download helper script for SelectiveMagnoViT
# Run this on the login node before preprocessing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/raw_dataset"

echo "=========================================="
echo "Dataset Download Helper"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo ""

# Function to download ImageNette
download_imagenette() {
    echo "Downloading ImageNette dataset..."
    echo "Size: ~1.4GB (320px version)"
    echo "Classes: 10 (subset of ImageNet)"
    echo ""

    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    # Download
    if [ ! -f "imagenette2-320.tgz" ]; then
        echo "Downloading imagenette2-320.tgz..."
        wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    else
        echo "imagenette2-320.tgz already exists, skipping download"
    fi

    # Extract
    echo "Extracting..."
    tar -xzf imagenette2-320.tgz

    # Reorganize
    echo "Reorganizing directory structure..."
    mv imagenette2-320/* .
    rmdir imagenette2-320

    # Verify
    echo ""
    echo "Verifying structure..."
    echo "Train classes:"
    ls -1 train/
    echo ""
    echo "Val classes:"
    ls -1 val/
    echo ""

    # Count images
    train_count=$(find train -name "*.JPEG" | wc -l)
    val_count=$(find val -name "*.JPEG" | wc -l)

    echo "Statistics:"
    echo "  Train images: $train_count"
    echo "  Val images: $val_count"
    echo "  Total: $((train_count + val_count))"
    echo ""
    echo "✓ ImageNette download complete!"
}

# Function to download informative-drawings pretrained model
download_pretrained_model() {
    echo ""
    echo "=========================================="
    echo "Downloading Informative-Drawings Pretrained Model"
    echo "=========================================="

    CHECKPOINT_DIR="$PROJECT_ROOT/third_party/informative-drawings/checkpoints"
    mkdir -p "$CHECKPOINT_DIR"
    cd "$CHECKPOINT_DIR"

    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown for Google Drive downloads..."
        pip install gdown
    fi

    # Download the model
    if [ ! -f "model.zip" ]; then
        echo "Downloading pretrained model from Google Drive..."
        echo "File ID: 1MIdHzecxz-z0uY3ARL_R40DlKcuQxiDk"
        gdown 1MIdHzecxz-z0uY3ARL_R40DlKcuQxiDk -O model.zip
    else
        echo "model.zip already exists, skipping download"
    fi

    # Unzip
    if [ ! -d "opensketch_style" ]; then
        echo "Extracting model..."
        unzip -q model.zip
        echo "✓ Model extracted to opensketch_style/"
    else
        echo "opensketch_style/ already exists, skipping extraction"
    fi

    # Verify
    if [ -d "opensketch_style" ]; then
        echo ""
        echo "Model files:"
        ls -lh opensketch_style/
        echo ""
        echo "✓ Pretrained model ready!"
    else
        echo "ERROR: Model extraction failed"
        exit 1
    fi
}

# Main menu
echo "What would you like to download?"
echo ""
echo "1) ImageNette dataset (~1.4GB)"
echo "2) Informative-drawings pretrained model (~100MB)"
echo "3) Both"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        download_imagenette
        ;;
    2)
        download_pretrained_model
        ;;
    3)
        download_imagenette
        download_pretrained_model
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify environment is set up:"
echo "   mamba activate drawings  # or: conda activate drawings"
echo ""
echo "2. Run tests to verify installation:"
echo "   sbatch slurm/submit_tests.sh"
echo ""
echo "3. Run preprocessing to generate line drawings:"
echo "   sbatch slurm/submit_preprocessing.sh"
echo ""
echo "4. Monitor preprocessing job:"
echo "   squeue -u \$USER"
echo "   tail -f logs/slurm/preprocess_<jobid>.out"
echo ""
