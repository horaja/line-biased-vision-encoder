import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SpatialThresholdSelector(nn.Module):
    """
    Selects patches using spatially-biased threshold mechanism.
    
    Combines patch density scores with spatial bias derived from the line drawing's
    center of gravity, selecting patches that are both dense and centrally located.
    
    Args:
        patch_percentage: Fraction of patches to select (0, 1]
        threshold: Minimum weighted score for automatic selection
        gaussian_std: Standard deviation for Gaussian spatial weighting

    Note these args must be treated like hyperparameters.
    """
    
    def __init__(
        self,
        patch_percentage: float,
        threshold: float = 0.3,
        gaussian_std: float = 0.25
    ):
        super().__init__()
        
        if not 0 < patch_percentage <= 1.0:
            raise ValueError(f"patch_percentage must be in (0, 1], got {patch_percentage}")
        
        self.patch_percentage = patch_percentage
        self.threshold = threshold
        self.gaussian_std = gaussian_std
    
    def _compute_center_of_gravity(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Compute the center of mass for each line drawing.
        
        Args:
            line_drawing: Tensor of shape (B, 1, H_LD, W_LD)
        
        Returns:
            Centers of shape (B, 2) with normalized coordinates [y, x] in [0, 1]
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
    
    def _create_gaussian_weights(
        self,
        num_patches_h: int,
        num_patches_w: int,
        centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Create 2D Gaussian weight maps centered on each batch's center of gravity.
        
        Args:
            num_patches_h: Number of patches in height dimension
            num_patches_w: Number of patches in width dimension
            centers: Centers of gravity of shape (B, 2)
        
        Returns:
            Gaussian weights of shape (B, num_patches_h * num_patches_w)

        Note: The args must be the height and width of color images.
        """
        B = centers.shape[0]
        device = centers.device
        
        # Create patch coordinate grid
        y_patch = torch.linspace(0, 1, num_patches_h, device=device)
        x_patch = torch.linspace(0, 1, num_patches_w, device=device)
        grid_y, grid_x = torch.meshgrid(y_patch, x_patch, indexing='ij')
        
        # Flatten grid coordinates (N, 2) - WHAT DOES THIS DO?
        grid_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        
        # Compute Gaussian weights for each batch item
        weights = []
        for b in range(B):
            center = centers[b].unsqueeze(0)  # (1, 2)
            distances_sq = ((grid_coords - center) ** 2).sum(dim=1)
            gaussian = torch.exp(-distances_sq / (2 * self.gaussian_std ** 2))
            weights.append(gaussian)
        
        return torch.stack(weights)  # (B, N)
    
    def _select_patch_indices(
        self,
        weighted_scores: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Select patch indices using threshold with fallback.
        
        Args:
            weighted_scores: Weighted patch scores of shape (B, N)
            k: Number of patches to select
        
        Returns:
            Selected indices of shape (B, k)
        """
        B, N = weighted_scores.shape
        indices_list = []
        
        for b in range(B):
            scores_b = weighted_scores[b]
            above_threshold = torch.where(scores_b > self.threshold)[0]
            
            if len(above_threshold) >= k:
                # Too many patches passed threshold - take top-k among them
                _, top_indices = torch.topk(scores_b[above_threshold], k=min(k, len(above_threshold)))
                selected = above_threshold[top_indices]
            elif len(above_threshold) > 0:
                # Some patches passed - take all and supplement with next best
                remaining_k = k - len(above_threshold)
                remaining_scores = scores_b.clone()
                remaining_scores[above_threshold] = -float('inf')
                _, remaining_indices = torch.topk(remaining_scores, k=remaining_k)
                selected = torch.cat([above_threshold, remaining_indices])
            else:
                # No patches passed threshold - fall back to pure top-k
                _, selected = torch.topk(scores_b, k=k)
            
            indices_list.append(selected)
        
        return torch.stack(indices_list)
    
    def forward(
        self,
        magno_patches: torch.Tensor,
        vit_positional_embedding: torch.Tensor,
        scores: torch.Tensor,
        line_drawing: torch.Tensor
    ) -> torch.Tensor:
        """
        Select patches based on spatial threshold strategy.
        
        Args:
            magno_patches: All patches from magno image, shape (B, N, D)
            vit_positional_embedding: Positional embeddings, shape (1, N+1, D)
            scores: Patch importance scores, shape (B, N)
            line_drawing: Line drawings, shape (B, 1, H, W)
        
        Returns:
            Selected patches with positional embeddings added, shape (B, k, D)
        """
        B, N, D = magno_patches.shape
        k = max(1, int(N * self.patch_percentage))
        
        # Compute spatial weights
        num_patches_side = int(np.sqrt(N))
        if num_patches_side ** 2 != N:
            raise ValueError(f"Number of patches {N} is not a perfect square")
        
        centers = self._compute_center_of_gravity(line_drawing)
        gaussian_weights = self._create_gaussian_weights(
            num_patches_side, num_patches_side, centers
        )
        
        # Combine scores with spatial bias
        weighted_scores = scores * gaussian_weights
        
        # Select patches
        selected_indices = self._select_patch_indices(weighted_scores, k)
        
        # Gather selected patches
        indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(magno_patches, 1, indices_expanded)
        
        # Gather positional embeddings (skip CLS token)
        pos_embed_patches = vit_positional_embedding[:, 1:, :]
        pos_embed_expanded = pos_embed_patches.expand(B, -1, -1)
        pos_embed_indices = selected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_pos_embed = torch.gather(pos_embed_expanded, 1, pos_embed_indices)
        
        # Add positional embeddings
        return selected_patches + selected_pos_embed