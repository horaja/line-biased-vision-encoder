import torch
import torch.nn as nn
from typing import Tuple


class PatchImportanceScorer(nn.Module):
    """
    Calculates importance scores for image patches based on line drawing density.
    
    Uses average pooling to efficiently compute the sum of pixel intensities
    within each patch region.
    
    Args:
        patch_size: Size of each patch (e.g., 4, 8, 16)
    """
    
    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        
        self.patch_size = patch_size
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
    
    def forward(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each patch.
        
        Args:
            line_drawing: Line drawing tensor of shape (B, 1, H, W)
                         with values in [0, 1], where higher values indicate
                         more line content
        
        Returns:
            Patch scores of shape (B, num_patches), where num_patches = (H/P) * (W/P)
        """
        if line_drawing.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {line_drawing.shape}")
        
        if line_drawing.size(1) != 1:
            raise ValueError(f"Expected single-channel input, got {line_drawing.size(1)} channels")
        
        # Compute average pooling
        avg_scores = self.pool(line_drawing)  # (B, 1, H/P, W/P)
        
        # Flatten to (B, num_patches)
        scores = avg_scores.flatten(start_dim=1)
        
        return scores
    
    def get_num_patches(self, img_size: int) -> int:
        """Calculate the number of patches for a given image size."""
        if img_size % self.patch_size != 0:
            raise ValueError(f"Image size {img_size} not divisible by patch size {self.patch_size}")
        return (img_size // self.patch_size) ** 2