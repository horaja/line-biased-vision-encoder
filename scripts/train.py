#!/usr/bin/env python3
"""
Training script for SelectiveMagnoViT model.
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
from selective_magno_vit.training.trainer import Trainer
from selective_magno_vit.data.dataset import get_dataloaders
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT
from selective_magno_vit.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train SelectiveMagnoViT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--color_dir",
        type=str,
        help="Override color image directory"
    )
    parser.add_argument(
        "--lines_dir",
        type=str,
        help="Override line drawing directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--patch_percentage",
        type=float,
        help="Override patch percentage"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    if args.color_dir:
        config.set('data.color_dir', args.color_dir)
    if args.lines_dir:
        config.set('data.lines_dir', args.lines_dir)
    if args.output_dir:
        config.set('output.checkpoint_dir', args.output_dir)
        run_name = Path(args.output_dir).name
        config.set('output.tensorboard_dir', f"logs/tensorboard/{run_name}")
    if args.patch_percentage:
        config.set('model.patch_percentage', args.patch_percentage)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.get('output.logs_dir'),
        log_level=config.get('logging.level', 'INFO')
    )
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    num_classes = train_loader.dataset.dataset.num_classes if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.num_classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = SelectiveMagnoViT(
        patch_percentage=config.get('model.patch_percentage'),
        num_classes=num_classes,
        color_img_size=config.get('model.color_img_size'),
        color_patch_size=config.get('model.color_patch_size'),
        ld_img_size=config.get('model.ld_img_size'),
        ld_patch_size=config.get('model.ld_patch_size'),
        vit_model_name=config.get('model.vit_model_name'),
        selector_config=config.get('model.selector')
    ).to(device)
    
    # # =========================================================
    # # START: Freeze layers for fine-tuning
    # # =========================================================
    # logger.info("Freezing backbone layers for fine-tuning...")
    
    # # 1. First, freeze the entire model
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # # 2. Unfreeze the Classification Head (New/Randomly initialized)
    # for param in model.vit.head.parameters():
    #     param.requires_grad = True
        
    # # 3. Unfreeze Positional Embeddings (New/Randomly initialized)
    # model.vit.pos_embed.requires_grad = True
    
    # # 4. Unfreeze Patch Embeddings (New/Randomly initialized)
    # # You must train this because you replaced the layer in __init__
    # for param in model.vit.patch_embed.parameters():
    #     param.requires_grad = True

    # # 5. Unfreeze Normalization Layer (Recommended)
    # # This helps adapt the frozen features to your specific dataset stats
    # for param in model.vit.norm.parameters():
    #     param.requires_grad = True
        
    # # Note: model.scorer and model.selector have no trainable parameters
    # # =========================================================
    # # END: Freeze layers
    # # =========================================================

    # Log the parameter counts to confirm
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Params: {total_params:,}")
    logger.info(f"Trainable Params: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    try:
        best_accuracy = trainer.train()
        logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pth")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()