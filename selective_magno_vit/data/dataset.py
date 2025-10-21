"""
Dataset classes for SelectiveMagnoViT.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Callable
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class ImageNetteDataset(Dataset):
    """
    Dataset for loading paired Magno images and Line Drawings.
    
    Args:
        magno_root: Root directory for Magno images
        lines_root: Root directory for Line Drawings
        split: Dataset split ('train' or 'val')
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        magno_root: str,
        lines_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.magno_root = Path(magno_root) / split
        self.lines_root = Path(lines_root) / split
        self.split = split
        self.transform = transform
        
        # Validate directories
        if not self.magno_root.exists():
            raise FileNotFoundError(f"Magno directory not found: {self.magno_root}")
        if not self.lines_root.exists():
            raise FileNotFoundError(f"Lines directory not found: {self.lines_root}")
        
        # Find all samples
        self.samples = self._find_samples()
        
        # Create class mapping
        self.class_names = sorted(set(sample['class_name'] for sample in self.samples))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        
        logger.info(f"Created {split} dataset with {len(self.samples)} samples and {len(self.class_names)} classes")
    
    def _find_samples(self):
        """Find all valid sample pairs."""
        samples = []
        
        # Walk through line drawings directory
        for line_path in sorted(self.lines_root.rglob('*_line.png')):
            # Extract class name and image name
            relative_path = line_path.relative_to(self.lines_root)
            class_name = relative_path.parent.name if relative_path.parent.name else relative_path.parts[0]
            
            # Construct corresponding magno path
            magno_name = line_path.stem.replace('_line', '_magno') + line_path.suffix
            magno_path = self.magno_root / relative_path.parent / magno_name
            
            # Verify both files exist
            if magno_path.exists():
                samples.append({
                    'magno_path': magno_path,
                    'line_path': line_path,
                    'class_name': class_name
                })
            else:
                logger.warning(f"Missing magno image for {line_path}")
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.lines_root}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load images
        magno_image = Image.open(sample['magno_path']).convert('RGB')
        line_drawing = Image.open(sample['line_path']).convert('L')
        
        # Apply transforms
        if self.transform:
            magno_image = self.transform(magno_image)
        else:
            magno_image = transforms.ToTensor()(magno_image)
        
        # Line drawing: invert and convert to tensor
        line_drawing = transforms.ToTensor()(line_drawing)
        line_drawing = 1.0 - line_drawing  # Invert so lines are white on black
        
        # Get label
        label = self.class_to_idx[sample['class_name']]
        
        return {
            'magno_image': magno_image,
            'line_drawing': line_drawing,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def get_dataloaders(config):
    """
    Create train and validation dataloaders from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config.get('model.img_size'),
            scale=tuple(config.get('training.augmentation.random_crop_scale'))
        ),
        transforms.RandomHorizontalFlip() if config.get('training.augmentation.random_horizontal_flip') else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(**config.get('training.augmentation.color_jitter')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.get('model.img_size'), config.get('model.img_size'))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageNetteDataset(
        magno_root=config.get('data.magno_dir'),
        lines_root=config.get('data.lines_dir'),
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNetteDataset(
        magno_root=config.get('data.magno_dir'),
        lines_root=config.get('data.lines_dir'),
        split='val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training.batch_size'),
        shuffle=True,
        num_workers=config.get('training.num_workers'),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training.batch_size'),
        shuffle=False,
        num_workers=config.get('training.num_workers'),
        pin_memory=True
    )
    
    return train_loader, val_loader