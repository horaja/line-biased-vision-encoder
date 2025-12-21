# Project: SelectiveMagnoViT

This project implements a Vision Transformer (ViT) that selectively processes image patches based on importance scores from line drawings, aiming to reduce computational cost.

## Core Technologies

*   **Language:** Python 3.10
*   **Framework:** PyTorch
*   **Model:** ViT from `timm`

## Project Structure

*   `selective_magno_vit/`: Main Python package.
    *   `models/`: Core `SelectiveMagnoViT` model.
    *   `data/`: Data loading and preprocessing.
    *   `training/`: `Trainer` class.
*   `scripts/`: Entry-point scripts for training, evaluation, etc.
*   `configs/`: YAML configuration files.
*   `slurm/`: Slurm submission scripts for cluster execution.

## Setup & Execution

### 1. Installation

This project uses `mamba` to manage a `conda` environment named `vla`.

```bash
# Create the environment and install dependencies
make install

# Activate the environment
mamba activate vla
```

### 2. Workflow

The primary workflows (preprocessing, training, evaluation) are executed via `make` commands that submit jobs to a Slurm cluster.

```bash
# Preprocess the data
make preprocess

# Start a training run
make train

# Evaluate a trained model
make evaluate
```

The behavior of these jobs is controlled by configuration files in the `configs/` directory.

## Development

*   **Formatting:** Use `black` for consistent code style.
    ```bash
    make format
    ```
*   **Linting:** Use `flake8` to check for code quality.
    ```bash
    make lint
    ```
*   **Testing:** Use `pytest` to run unit tests.
    ```bash
    make test
    ```

## Code Style

*   **Philosophy:** Write idiomatic but simple Python.
*   **Clarity:** Prioritize conciseness and simplicity to create clean, readable code.
*   **Comments:** Use comments sparingly. Focus on *why* something is done, not *what* is being done.

## Agent Instructions

*   Do not delete any major functionality in the codebase unless explicitly asked to do so.
