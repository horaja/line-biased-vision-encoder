import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def _compute_cog(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Computes the center of mass of the line drawings of each image
        
        :param self
        :param line_drawing: Line Drawings of shape (B, 1, H, W)
        :type line_drawing: torch.Tensor
        :return: centers of shape (B, 2) ST each (y,x) in [0,1]
        :rtype: torch.tensor
        """
        B, _, H, W = line_drawing.shape
        device = line_drawing.device
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        
        # Compute weighted sums
        total_mass = line_drawing.sum(dim=(2, 3), keepdim=True) + 1e-8  # (B, 1, 1, 1)
        weighted_y = (line_drawing * y_coords).sum(dim=(2, 3))  # (B, 1)
        weighted_x = (line_drawing * x_coords).sum(dim=(2, 3))  # (B, 1)
        
        # Normalize by total mass
        cog_y = weighted_y / total_mass.squeeze(2).squeeze(2)  # (B, 1)
        cog_x = weighted_x / total_mass.squeeze(2).squeeze(2)  # (B, 1)
        
        # Stack to (B, 2)
        return torch.cat([cog_y, cog_x], dim=1)
    
    def forward(self, line_drawing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute importance scores for each patch.
        
        Args:
            line_drawing: Line drawing tensor of shape (B, 1, H, W)
                         with values in [0, 1], where higher values indicate
                         more line content
        
        Returns:
            Patch scores of shape (B, num_patches), where num_patches = (H/P) * (W/P),
            and centers of gravity of shape (B, 2).
        """
        if line_drawing.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {line_drawing.shape}")
        
        if line_drawing.size(1) != 1:
            raise ValueError(f"Expected single-channel input, got {line_drawing.size(1)} channels")
        
        # Compute average pooling
        avg_scores = self.pool(line_drawing)  # (B, 1, H/P, W/P)
        
        # Flatten to (B, num_patches)
        raw_scores = avg_scores.flatten(start_dim=1)

        # Create a probability distribution via softmax
        scores = F.softmax(raw_scores, dim=-1)

        # Compute center of gravity
        cog = self._compute_cog(line_drawing)
        
        return scores, cog
