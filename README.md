[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Line-Guided Patch Selection ViT

TL;DR: Line drawings guide a Vision Transformer to keep only the most informative patches, cutting compute while matching or beating full-patch baselines. Precomputed runs and visuals live in `results/`, so you can review in 60 seconds.

## Quickstart (60s)
```bash
mamba env create -f environment.yml  # creates the vla env
mamba activate vla
git submodule update --init --recursive  # pulls informative-drawings
pip install -e .
```

<details>
<summary>Smoke test (no data needed)</summary>

```bash
python - <<'PY'
import torch
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT

model = SelectiveMagnoViT(
    patch_percentage=0.4,
    num_classes=10,
    color_img_size=256,
    ld_img_size=64,
    pretrained=False
)
model.eval()
color = torch.randn(1, 3, 256, 256)
lines = torch.randn(1, 1, 64, 64)
with torch.no_grad():
    logits = model(color, lines)
print("Logits shape:", logits.shape)
print("Selected patches:", model.get_num_selected_patches())
PY
```

</details>

## Results (ImageNet-10 val)
| Setting | Top-1 | Top-5 | GFLOPs | Artifacts |
| --- | --- | --- | --- | --- |
| Full patch budget (~all patches) | 0.734 | 0.964 | 1.408 | Confusion/per-class: `results/smart/eval_best_model_val_20251224_033043/visualizations/`; tradeoff plots: `results/fig_accuracy_vs_gflops.png`, `results/fig_gflops_scaling.png` |
| Line-guided selective (low compute) | **0.758** | **0.968** | **0.448** | Patch selection + confusion: `results/smart/eval_best_model_val_20251224_035354/visualizations/` |

Visuals at a glance:

| Tradeoff curve | Selected patches (0.45 GFLOPs) | Confusion (selective) |
| --- | --- | --- |
| ![](assets/combined_tradeoff_figure.png) | ![](assets/patch_selection_sample_000.png) | ![](assets/confusion_matrix_normalized.png) |

## How it works (3 steps)
```mermaid
graph LR
    A[Line Drawing (Magno Stream)] --> S[Selector]
    B[Color Image (Parvo Stream)] --> E[Patch Embed]
    E --> S
    S --> V[ViT Blocks]
    V --> H[Classifier Head / Logits]
```

## Train / evaluate locally (needs data)
- Preprocess (requires GPU and the `third_party/informative-drawings` submodule):
  ```bash
  python scripts/preprocess.py \
    --config configs/base_config.yml \
    --raw_data_root <raw_imagenette_or_imagenet100> \
    --preprocessed_root data/preprocessed
  ```
- Train (example, patch_percentage 0.4):
  ```bash
  python scripts/train.py \
    --config configs/base_config.yml \
    --color_dir data/preprocessed/color_images \
    --lines_dir data/preprocessed/line_drawings \
    --output_dir checkpoints/local_p0.4 \
    --patch_percentage 0.4 \
    --epochs 5 \
    --batch_size 16
  ```
- Evaluate a checkpoint:
  ```bash
  python scripts/evaluate.py \
    --config configs/base_config.yml \
    --checkpoint checkpoints/local_p0.4/best_model.pth \
    --color_dir data/preprocessed/color_images \
    --lines_dir data/preprocessed/line_drawings \
    --split val \
    --output_dir results/local_eval_p0.4 \
    --analyze_patches
  ```
SLURM equivalents remain under `slurm/` if you want cluster runs.

## Repo map
```text
.
├─ selective_magno_vit/
│  ├─ models/              # line-guided ViT + selector modules
│  ├─ data/                # datasets and preprocessing
│  ├─ training/            # trainer and callbacks
│  └─ evaluation/          # evaluator and visualizer
├─ scripts/                # train/eval/preprocess/visualize entrypoints
├─ configs/                # YAML configs
├─ results/                # precomputed plots, metrics, visuals (git-ignored)
├─ assets/                 # checked-in copies of featured visuals
├─ slurm/                  # cluster submission scripts
├─ third_party/            # submodules (informative-drawings)
└─ docs/                   # dev notes
```

## Citation
```
@misc{line_guided_patch_selection_2025,
  title  = {Line-Guided Patch Selection for Vision Transformers},
  author = {Husain, Tianqin and collaborators},
  year   = {2025},
  note   = {GitHub repository},
}
```
