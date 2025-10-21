"""
Custom data transformations for SelectiveMagnoViT.

Since the dataset.py already uses torchvision.transforms.Compose for standard
transformations, this module is reserved for any custom transforms specific
to the magno/line drawing preprocessing.
"""

import torch
import torchvision.transforms as transforms
from typing import Tuple
from PIL import Image


class PairedImageTransform:
    """
    Apply the same random transformations to both magno image and line drawing.

    This ensures that spatial augmentations (like crops and flips) are
    applied consistently to both inputs.
    """

    def __init__(
        self,
        img_size: int = 224,
        random_crop: bool = True,
        random_flip: bool = True
    ):
        """
        Args:
            img_size: Target image size
            random_crop: Whether to apply random cropping
            random_flip: Whether to apply random horizontal flip
        """
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __call__(
        self,
        magno_image: Image.Image,
        line_drawing: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations to both images.

        Args:
            magno_image: RGB magno image
            line_drawing: Grayscale line drawing

        Returns:
            Tuple of (transformed_magno, transformed_line)
        """
        # Get random crop parameters if enabled
        if self.random_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                magno_image,
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0)  # Keep square aspect ratio
            )
            magno_image = transforms.functional.resized_crop(
                magno_image, i, j, h, w, (self.img_size, self.img_size)
            )
            line_drawing = transforms.functional.resized_crop(
                line_drawing, i, j, h, w, (self.img_size, self.img_size)
            )
        else:
            magno_image = transforms.functional.resize(magno_image, (self.img_size, self.img_size))
            line_drawing = transforms.functional.resize(line_drawing, (self.img_size, self.img_size))

        # Apply random horizontal flip to both
        if self.random_flip and torch.rand(1).item() > 0.5:
            magno_image = transforms.functional.hflip(magno_image)
            line_drawing = transforms.functional.hflip(line_drawing)

        # Convert to tensors
        magno_tensor = transforms.functional.to_tensor(magno_image)
        line_tensor = transforms.functional.to_tensor(line_drawing)

        # Normalize magno image (standard ImageNet normalization)
        magno_tensor = transforms.functional.normalize(
            magno_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Invert line drawing so lines are white on black background
        line_tensor = 1.0 - line_tensor

        return magno_tensor, line_tensor


class LineDrawingInvert:
    """
    Inverts line drawing so lines are white (1.0) on black (0.0) background.

    This is useful because the informative-drawings model typically outputs
    black lines on white background, but our scorer expects white lines.
    """

    def __call__(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Invert the line drawing.

        Args:
            line_drawing: Tensor of shape (1, H, W) or (H, W)

        Returns:
            Inverted tensor
        """
        return 1.0 - line_drawing


class NormalizeLineDrawing:
    """
    Normalizes line drawing to [0, 1] range.

    Useful if line drawings have been saved with different value ranges.
    """

    def __call__(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Normalize line drawing to [0, 1].

        Args:
            line_drawing: Input tensor

        Returns:
            Normalized tensor
        """
        min_val = line_drawing.min()
        max_val = line_drawing.max()

        if max_val - min_val > 0:
            return (line_drawing - min_val) / (max_val - min_val)
        else:
            return line_drawing


def get_train_transforms(img_size: int = 224):
    """
    Get standard training transformations for magno images.

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: int = 224):
    """
    Get standard validation transformations for magno images.

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_line_drawing_transforms():
    """
    Get standard transformations for line drawings.

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        LineDrawingInvert()  # Invert so lines are white
    ])
