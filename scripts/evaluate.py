#!/usr/bin/env python3
"""
Evaluation script for SelectiveMagnoViT model.
Updated for Standard ImageNet-100 workflow.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selective_magno_vit.utils.config import Config
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT
# CHANGED: Import get_dataloaders instead of specific Dataset class
from selective_magno_vit.data.dataset import get_dataloaders
from selective_magno_vit.evaluation.evaluator import ModelEvaluator
from selective_magno_vit.utils.logging import setup_logging
from selective_magno_vit.utils.checkpointing import load_checkpoint
from selective_magno_vit.utils.config_validation import validate_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SelectiveMagnoViT")
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
        "--split",
        type=str,
        default="test", # Default to held-out test set
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (train=subset, val=subset, test=held-out)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--patch_percentage",
        type=float,
        help="Override patch percentage used for patch selection"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--analyze_patches",
        action="store_true",
        help="Analyze patch selection behavior"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model predictions"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = Config(args.config)

    # Override with command line arguments
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.patch_percentage is not None:
        config.set('model.patch_percentage', args.patch_percentage)
    if args.output_dir:
        config.set('output.results_dir', args.output_dir)

    validate_config(config)

    # Setup logging
    logger = setup_logging(
        log_dir=config.get('output.logs_dir'),
        log_level=config.get('logging.level', 'INFO')
    )

    logger.info(f"Starting evaluation on {args.split} split")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # CHANGED: Use get_dataloaders to ensure consistency with training splits
    # get_dataloaders returns (train_loader, val_loader, test_loader)
    logger.info("Initializing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Select the requested loader
    if args.split == 'train':
        dataloader = train_loader
        logger.info("Selected: Training Subset")
    elif args.split == 'val':
        dataloader = val_loader
        logger.info("Selected: Validation Subset (Model Selection Set)")
    else: # test
        dataloader = test_loader
        logger.info("Selected: Held-Out Test Set (Raw 'val' folder)")

    # Access the underlying dataset to get class info
    # Handle Subset wrappers if present
    if hasattr(dataloader.dataset, 'dataset'):
        full_ds = dataloader.dataset.dataset
    else:
        full_ds = dataloader.dataset
        
    num_classes = full_ds.num_classes
    class_names = full_ds.classes
    
    logger.info(f"Dataset: {len(dataloader.dataset)} samples, {num_classes} classes")

    # Create model
    model = SelectiveMagnoViT(
        patch_percentage=config.get('model.patch_percentage'),
        num_classes=num_classes,
        color_img_size=config.get('model.color_img_size'),
        color_patch_size=config.get('model.color_patch_size'),
        ld_img_size=config.get('model.ld_img_size', 64),
        ld_patch_size=config.get('model.ld_patch_size', 4),
        vit_model_name=config.get('model.vit_model_name'),
        selector_config=config.get('model.selector')
    ).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=class_names
    )

    # Run evaluation
    results = evaluator.evaluate(
        dataloader,
        compute_confusion=True,
        compute_topk=True,
        k=5
    )

    # Analyze patch selection if requested
    if args.analyze_patches:
        logger.info("\nAnalyzing patch selection...")
        patch_stats = evaluator.evaluate_patch_selection(dataloader, num_samples=100)
        results['patch_selection_stats'] = patch_stats

    # Save predictions if requested
    if args.save_predictions:
        logger.info("\nSaving predictions...")
        predictions, probabilities = evaluator.predict(dataloader, return_probabilities=True)
        results['predictions'] = predictions

    # Save results
    results_dir = Path(config.get('output.results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    results_file = results_dir / f"evaluation_{checkpoint_name}_{args.split}.json"

    evaluator.save_results(results, results_file)

    logger.info(f"\nEvaluation complete!")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Final Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
