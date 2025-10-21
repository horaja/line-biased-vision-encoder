#!/usr/bin/env python3
"""
Visualization script for SelectiveMagnoViT results.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selective_magno_vit.utils.config import Config
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT
from selective_magno_vit.data.dataset import ImageNetteDataset
from selective_magno_vit.evaluation.visualizer import (
    PatchSelectionVisualizer,
    plot_confusion_matrix,
    plot_per_class_performance
)
from selective_magno_vit.utils.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SelectiveMagnoViT results")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--magno_dir",
        type=str,
        help="Override magno image directory"
    )
    parser.add_argument(
        "--lines_dir",
        type=str,
        help="Override line drawing directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of patch selection samples to visualize"
    )
    parser.add_argument(
        "--visualize_patches",
        action="store_true",
        help="Visualize patch selection"
    )
    parser.add_argument(
        "--visualize_confusion",
        action="store_true",
        help="Visualize confusion matrix"
    )
    parser.add_argument(
        "--visualize_performance",
        action="store_true",
        help="Visualize per-class performance"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If --all is specified, enable all visualizations
    if args.all:
        args.visualize_patches = True
        args.visualize_confusion = True
        args.visualize_performance = True

    # Load configuration
    config = Config(args.config)

    # Override with command line arguments
    if args.magno_dir:
        config.set('data.magno_dir', args.magno_dir)
    if args.lines_dir:
        config.set('data.lines_dir', args.lines_dir)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    val_transform = transforms.Compose([
        transforms.Resize((config.get('model.img_size'), config.get('model.img_size'))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNetteDataset(
        magno_root=config.get('data.magno_dir'),
        lines_root=config.get('data.lines_dir'),
        split=args.split,
        transform=val_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Use batch size 1 for visualization
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = dataset.num_classes
    print(f"Dataset: {len(dataset)} samples, {num_classes} classes")

    # Create model
    model = SelectiveMagnoViT(
        patch_percentage=config.get('model.patch_percentage'),
        num_classes=num_classes,
        img_size=config.get('model.img_size'),
        patch_size=config.get('model.patch_size'),
        vit_model_name=config.get('model.vit_model_name'),
        selector_config=config.get('model.selector')
    ).to(device)

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Visualize patch selection
    if args.visualize_patches:
        print(f"\nGenerating patch selection visualizations ({args.num_samples} samples)...")
        visualizer = PatchSelectionVisualizer(model, device)
        patch_dir = output_dir / "patch_selection"
        visualizer.visualize_multiple_samples(
            dataloader,
            num_samples=args.num_samples,
            save_dir=patch_dir
        )
        print(f"Saved patch visualizations to {patch_dir}")

    # Load results if provided
    results = None
    if args.results_file:
        print(f"\nLoading results from {args.results_file}...")
        with open(args.results_file, 'r') as f:
            results = json.load(f)

    # Visualize confusion matrix
    if args.visualize_confusion and results and 'confusion_matrix' in results:
        print("\nGenerating confusion matrix visualization...")
        confusion_matrix = np.array(results['confusion_matrix'])

        # Non-normalized
        plot_confusion_matrix(
            confusion_matrix,
            dataset.class_names,
            save_path=output_dir / "confusion_matrix.png",
            show=False,
            normalize=False
        )

        # Normalized
        plot_confusion_matrix(
            confusion_matrix,
            dataset.class_names,
            save_path=output_dir / "confusion_matrix_normalized.png",
            show=False,
            normalize=True
        )
        print(f"Saved confusion matrices to {output_dir}")

    # Visualize per-class performance
    if args.visualize_performance and results and 'per_class_metrics' in results:
        print("\nGenerating per-class performance visualization...")
        per_class_metrics = results['per_class_metrics']

        class_names = [m['class'] for m in per_class_metrics]
        accuracies = [m['accuracy'] for m in per_class_metrics]
        f1_scores = [m['f1_score'] for m in per_class_metrics]

        plot_per_class_performance(
            class_names,
            accuracies,
            f1_scores,
            save_path=output_dir / "per_class_performance.png",
            show=False
        )
        print(f"Saved per-class performance to {output_dir}")

    print("\nVisualization complete!")
    print(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
