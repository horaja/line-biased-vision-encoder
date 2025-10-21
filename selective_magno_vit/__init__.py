"""
SelectiveMagnoViT: Efficient Vision Transformer with Line Drawing-Guided Patch Selection
"""

__version__ = "0.1.0"

from .models.selective_vit import SelectiveMagnoViT
from .data.dataset import ImageNetteDataset
from .training.trainer import Trainer
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    "SelectiveMagnoViT",
    "ImageNetteDataset",
    "Trainer",
    "ModelEvaluator",
]