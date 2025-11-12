#!/usr/bin/env python3
"""
Preprocessing script for generating line drawings and magno images.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selective_magno_vit.utils.config import Config
from selective_magno_vit.data.preprocessing import preprocess_imagenette
from selective_magno_vit.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset for SelectiveMagnoViT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--raw_data_root",
        type=str,
        help="Override raw data directory"
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        help="Override preprocessed data directory"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    raw_data_root = args.raw_data_root or config.get('data.raw_root')
    preprocessed_root = args.preprocessed_root or config.get('data.preprocessed_root')
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.get('output.logs_dir'),
        log_level=config.get('logging.level', 'INFO')
    )
    
    logger.info("Starting preprocessing pipeline")
    logger.info(f"Raw data: {raw_data_root}")
    logger.info(f"Output: {preprocessed_root}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Image sizes - Magno: {config.get('preprocessing.magno_size')}x{config.get('preprocessing.magno_size')}, Color: {config.get('preprocessing.color_size')}x{config.get('preprocessing.color_size')}")
    
    try:
        preprocess_imagenette(
            raw_data_root=raw_data_root,
            preprocessed_root=preprocessed_root,
            magno_size=config.get('preprocessing.magno_size'),
            color_size=config.get('preprocessing.color_size'),
            splits=args.splits
        )
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()