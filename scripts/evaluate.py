#!/usr/bin/env python3
"""
Evaluation script for SelectiveMagnoViT model.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selective_magno_vit.utils.config import Config
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT
from selective_magno_vit.data.dataset import ImageNetteDataset
from selective_magno_vit.evaluation.evaluator import ModelEvaluator
from selective_magno_vit.utils.logging import setup_logging
from selective_magno_vit.utils.checkpointing import load_checkpoint


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
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
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
    if args.magno_dir:
        config.set('data.magno_dir', args.magno_dir)
    if args.lines_dir:
        config.set('data.lines_dir', args.lines_dir)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.output_dir:
        config.set('output.results_dir', args.output_dir)

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
        batch_size=config.get('training.batch_size', 32),
        shuffle=False,
        num_workers=config.get('training.num_workers', 4),
        pin_memory=True
    )

    num_classes = dataset.num_classes
    logger.info(f"Dataset: {len(dataset)} samples, {num_classes} classes")

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
    logger.info("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Print model info
    model_info = model.get_model_info()
    logger.info("\nModel Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=dataset.class_names
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
        # Note: probabilities are numpy arrays and may need special handling for JSON

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
