.PHONY: help install test clean lint format preprocess train evaluate

help:
	@echo "SelectiveMagnoViT Makefile"
	@echo "=========================="
	@echo "install      - Install the package and dependencies"
	@echo "test         - Run unit tests"
	@echo "lint         - Run code linting"
	@echo "format       - Format code with black"
	@echo "clean        - Remove generated files"
	@echo "preprocess   - Run preprocessing pipeline"
	@echo "train        - Submit training job to SLURM"
	@echo "evaluate     - Submit evaluation job to SLURM"

install:
	@echo "Creating environment with mamba (or conda if mamba unavailable)..."
	@command -v mamba >/dev/null 2>&1 && mamba env create -f environment.yml || conda env create -f environment.yml
	@command -v mamba >/dev/null 2>&1 && mamba run -n drawings pip install -e . || conda run -n drawings pip install -e .
	git submodule update --init --recursive
	@echo ""
	@echo "Installation complete! Activate with: mamba activate drawings (or conda activate drawings)"

test:
	pytest tests/ -v

lint:
	flake8 selective_magno_vit/ scripts/

format:
	black selective_magno_vit/ scripts/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

preprocess:
	sbatch slurm/submit_preprocessing.sh

train:
	sbatch slurm/submit_training.sh

evaluate:
	sbatch slurm/submit_evaluation.sh