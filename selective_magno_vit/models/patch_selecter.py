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
        gaussian_std: Standard deviation for Gaussian spatial weighting

    Note these args must be treated like hyperparameters.
    """
    
    def __init__(
        self,
        patch_percentage: float,
        gaussian_std: float = 0.25
    ):
        super().__init__()
        
        if not 0 < patch_percentage <= 1.0:
            raise ValueError(f"patch_percentage must be in (0, 1], got {patch_percentage}")
        
        self.patch_percentage = patch_percentage
        self.gaussian_std = gaussian_std
    
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
        
        # Flatten grid coordinates (N, 2) - row-major flattening 
        grid_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        
        # Compute Gaussian weights for each batch item
        weights = []
        for b in range(B):
            center = centers[b].unsqueeze(0)  # (1, 2)
            distances_sq = ((grid_coords - center) ** 2).sum(dim=1)
            gaussian = torch.exp(-distances_sq / (2 * self.gaussian_std ** 2))
            weights.append(gaussian)
        
        weights = torch.stack(weights)  # (B, N)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8) # Normalize per batch
        
        return weights
    
    def _project_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Projects patch indices from low-resolution LD grid to high-res color image grid.
        
        :param self
        :param indices: selected indices of shape (B, k)
        :type indices: torch.Tensor
        :return: selected indices of shape (B, k), where each set of indices are a collection of new shape indices.
        :rtype: torch.Tensor

        TODO: Implement more general logic
        1. Calculate grid dimensions for source and target
        2. source index -> 2D grid coords -> normalized grid coords (between 0,0 and 1,1) -> target 2d grid coords -> target index
        """
        projected_indices = indices
        return projected_indices

    def get_indices(self, scores : torch.Tensor, centers : torch.Tensor) -> torch.Tensor:
        """
        Generate selected indices using multinomial sampling and spatial bias.
        Expose helper function for visualization and consistency.
        
        :param self
        :param scores: patch importance scores of shape (B, N)
        :type scores: torch.Tensor
        :param centers: center of gravities of items in batch of shape (B, 2)
        :type centers: torch.Tensor
        :return: projected selected indices of shape (B, k)
        :rtype: Any
        """
        B, N = scores.shape()
        k = max(1, int(N * self.patch_percentage)) # k : number of patches to select
        
        # Compute spatial weights
        num_patches_side = int(np.sqrt(N))
        if num_patches_side ** 2 != N:
            raise ValueError(f"Number of patches {N} is not a perfect square")
        
        gaussian_weights = self._create_gaussian_weights(
            num_patches_side, num_patches_side, centers
        ) # spatial prior probability distribution

        # Combine scores with spatial bias
        joint = scores * gaussian_weights # (B, N)
        joint = joint / (joint.sum(dim=1, keepdim=True) + 1e-8) # renormalize per batch

        # Select patches
        selected_indices = torch.multinomial(joint, num_samples=k, replacement=False) # (B, k)

        return self._project_indices(selected_indices)

    
    def forward(
        self,
        color_patches: torch.Tensor,
        vit_positional_embedding: torch.Tensor,
        scores: torch.Tensor,
        centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Select patches based on spatial threshold strategy.
        
        Args:
            color_patches: All patches from color image, shape (B, N, D)
            vit_positional_embedding: Positional embeddings, shape (1, N+1, D)
            scores: Patch importance scores, shape (B, N)
            centers: center of gravities, shape (B, 2)
        
        Returns:
            Selected patches with positional embeddings added, shape (B, k, D)
        """
        B, _, D = color_patches.shape
        
        # Get projected selected indices
        projected_indices = self.get_indices(scores, centers)

        # Gather selected patches
        indices_expanded = projected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(color_patches, 1, indices_expanded)
        
        # Gather positional embeddings (skip CLS token)
        pos_embed_patches = vit_positional_embedding[:, 1:, :]
        pos_embed_expanded = pos_embed_patches.expand(B, -1, -1)
        pos_embed_indices = projected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_pos_embed = torch.gather(pos_embed_expanded, 1, pos_embed_indices)
        
        # Add positional embeddings
        return selected_patches + selected_pos_embed