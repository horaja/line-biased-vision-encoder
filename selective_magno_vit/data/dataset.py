"""
Dataset classes for SelectiveMagnoViT (Standard ImageNet-style support).
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, datasets
from PIL import Image

logger = logging.getLogger(__name__)

class StandardImageDataset(datasets.ImageFolder):
    """
    Standard ImageFolder dataset wrapper for SelectiveMagnoViT.
    Returns None for line_drawing to support random/color-only inference.
    """
    def __getitem__(self, index: int) -> Dict[str, any]:
        # Use parent ImageFolder logic to get image and label
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Return dict matching expected model input, with line_drawing as None
        return {
            'color_image': sample,
            'line_drawing': None,  # Signal to model to use random selection
            'label': torch.tensor(target, dtype=torch.long)
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)

def collate_none_safe(batch):
    """
    Custom collate function that handles None values by returning None 
    instead of crashing.
    """
    elem = batch[0]
    
    # If the element is a dictionary, process each key
    if isinstance(elem, dict):
        return {key: collate_none_safe([d[key] for d in batch]) for key in elem}
    
    # If the element is None, return None (handles the line_drawing case)
    if elem is None:
        return None
        
    # Otherwise use default behavior
    return default_collate(batch)

def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders from a standard ImageNet directory structure.
    
    Structure assumed:
        color_dir/
            train/
            val/
            
    Logic:
        - color_dir/val -> Held-out TEST set
        - color_dir/train -> Split into TRAIN and VALIDATION sets
    """
    # 1. Define Transforms
    img_size = config.get('model.color_img_size', 224)
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=config.get('training.augmentation.random_crop_scale', (0.08, 1.0))),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(**config.get('training.augmentation.color_jitter', {})),
        transforms.ToTensor(),
        normalize,
    ])

    # Transform for Validation and Test (Deterministic)
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)), # Resize slightly larger than target
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    data_root = Path(config.get('data.color_dir'))
    train_dir = data_root / 'train'
    val_dir = data_root / 'val' # This becomes our held-out TEST set

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Could not find 'train' or 'val' directories in {data_root}")

    # 2. Setup Datasets
    # A. Held-out Test Set
    test_dataset = StandardImageDataset(
        root=str(val_dir),
        transform=eval_transform
    )

    # B. Training Source (to be split)
    full_train_dataset = StandardImageDataset(
        root=str(train_dir),
        transform=train_transform 
    )
    
    # We need a separate view of the data for validation to apply the correct transforms (no jitter/flip)
    # Since we are splitting indices, we can create a second dataset object pointing to the same data
    full_train_for_val = StandardImageDataset(
        root=str(train_dir),
        transform=eval_transform
    )

    # 3. Create Train/Val Split
    # Example: 90% Train, 10% Validation
    total_train_samples = len(full_train_dataset)
    train_size = int(0.9 * total_train_samples)
    val_size = total_train_samples - train_size
    
    # Generate split indices
    generator = torch.Generator().manual_seed(config.get('experiment.seed', 42))
    train_subset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_subset = random_split(full_train_for_val, [train_size, val_size], generator=generator)
    
    # 4. Create Loaders
    num_workers = config.get('training.num_workers', 4)
    batch_size = config.get('training.batch_size', 32)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_none_safe
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_none_safe
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_none_safe
    )
    
    logger.info(f"Dataset Split -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test (Held-out): {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader