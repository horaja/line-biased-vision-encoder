"""
SelectiveMagnoViT: Vision Transformer with line drawing-guided selective patch processing.

This module implements the main model that combines:
1. Patch importance scoring based on line drawings
2. Spatial threshold-based patch selection
3. Vision Transformer processing on selected patches
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict

from .patch_scorer import PatchImportanceScorer
from .patch_selecter import SpatialThresholdSelector


class SelectiveMagnoViT(nn.Module):
    """
    SelectiveMagnoViT: A Vision Transformer that selectively processes patches
    based on importance scores derived from line drawings.

    The model works in three stages:
    1. Score patches using line drawing density (PatchImportanceScorer)
    2. Select important patches using spatial threshold strategy (SpatialThresholdSelector)
    3. Process selected patches through a Vision Transformer

    Args:
        patch_percentage: Fraction of patches to select (0, 1]
        num_classes: Number of output classes for classification
        color_img_size: COLOR input image size (assumes square images)
        color_patch_size: Size of each COLOR patch
        ld_img_size: LINE DRAWING input image size (assumes square images)
        ld_patch_size: Size of each LINE DRAWING patch
        vit_model_name: Name of the pre-trained ViT model from timm
        selector_config: Configuration dict for the patch selector
        embed_dim: Embedding dimension (if None, uses ViT default)
        pretrained: Whether to use pretrained ViT weights
    """

    def __init__(
        self,
        patch_percentage: float = 0.4,
        num_classes: int = 10,
        color_img_size: int = 256,
        color_patch_size: int = 16,
        ld_img_size: int = 64,
        ld_patch_size: int = 4,
        vit_model_name: str = 'vit_tiny_patch16_224.augreg_in21k',
        selector_config: Optional[Dict] = None,
        embed_dim: Optional[int] = None,
        pretrained: bool = True
    ):
        super().__init__()

        # Validate inputs
        if not 0 < patch_percentage <= 1.0:
            raise ValueError(f"patch_percentage must be in (0, 1], got {patch_percentage}")
        if color_img_size % color_patch_size != 0:
            raise ValueError(f"color_img_size ({color_img_size}) must be divisible by color_patch_size ({color_patch_size})")
        if ld_img_size % ld_patch_size != 0:
            raise ValueError(f"ld_img_size ({ld_img_size}) must be divisible by ld_patch_size ({ld_patch_size})")

        # Store configuration
        self.patch_percentage = patch_percentage
        self.num_classes = num_classes
        self.color_img_size = color_img_size
        self.color_patch_size = color_patch_size
        self.ld_img_size = ld_img_size
        self.ld_patch_size = ld_patch_size
        self.vit_model_name = vit_model_name

        # Load ViT backbone from timm
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained)

        # Get embedding dimension from loaded model
        if embed_dim is None:
            embed_dim = self.vit.embed_dim
        self.embed_dim = embed_dim

        # Replace patch embedding layer for custom image size
        self.vit.patch_embed = timm.models.vision_transformer.PatchEmbed(
            img_size=color_img_size,
            patch_size=color_patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        # Update positional embeddings for new number of patches
        num_patches = self.vit.patch_embed.num_patches
        self.vit.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)  # +1 for CLS token
        )
        nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)

        # Replace classifier head for custom number of classes
        self.vit.head = nn.Linear(embed_dim, num_classes)

        # Initialize custom modules for patch selection
        self.scorer = PatchImportanceScorer(patch_size=ld_patch_size)

        # Setup selector with config
        if selector_config is None:
            selector_config = {}
        self.selector = SpatialThresholdSelector(
            patch_percentage=patch_percentage,
            gaussian_std=selector_config.get('gaussian_std', 0.25),
            strategy=selector_config.get('strategy', 'smart')
        )
        print(f"Strategy: {selector_config.get('strategy', 'smart')}")

        # Store metadata
        self.num_patches = num_patches

    def forward(self, color_image: torch.Tensor, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            color_image: Batch of color images of shape (B, 3, H, W)
                        These are the actual images to be processed
            line_drawing: Batch of line drawings of shape (B, 1, H, W)
                         Used to determine which patches are important

        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Step 1: Score patches based on line drawing density
        patch_scores, centers = self.scorer(line_drawing)  # (B, num_patches)

        # Step 2: Extract all patches from the color image
        all_patches = self.vit.patch_embed(color_image)  # (B, num_patches, embed_dim)

        # Step 3: Select important patches using spatial threshold strategy
        # This adds positional embeddings to the selected patches
        selected_patches = self.selector(
            all_patches,
            self.vit.pos_embed,
            patch_scores,
            centers
        )  # (B, k, embed_dim) where k = num_patches * patch_percentage

        # Step 4: Prepare [CLS] token with its positional embedding
        cls_token_with_pos = self.vit.cls_token + self.vit.pos_embed[:, :1, :]

        # Step 5: Combine [CLS] token with selected patches
        batch_size = color_image.shape[0]
        full_sequence = torch.cat([
            cls_token_with_pos.expand(batch_size, -1, -1),  # (B, 1, embed_dim)
            selected_patches  # (B, k, embed_dim)
        ], dim=1)  # (B, k+1, embed_dim)

        # Step 6: Apply dropout to the full sequence
        full_sequence = self.vit.pos_drop(full_sequence)

        # Step 7: Process through transformer blocks
        x = self.vit.blocks(full_sequence)
        x = self.vit.norm(x)

        # Step 8: Extract [CLS] token output for classification
        cls_output = x[:, 0]  # (B, embed_dim)

        # Step 9: Final classification head
        logits = self.vit.head(cls_output)  # (B, num_classes)

        return logits

    # TODO: Change to track currently sampled patches
    @torch.no_grad()
    def get_selected_patch_indices(self, line_drawing: torch.Tensor) -> torch.Tensor:
        """
        Get indices of selected patches for visualization purposes.

        This method is useful for understanding which patches the model
        considers important for a given line drawing.

        Args:
            line_drawing: Line drawing tensor of shape (B, 1, H, W)

        Returns:
            Indices of selected patches of shape (B, k) where k is the number
            of selected patches
        """
        # Score patches
        patch_scores, centers = self.scorer(line_drawing)

        # Get the multinomial sampled indices
        indices = self.selector.get_indices(patch_scores, centers)

        return indices

    def get_num_selected_patches(self) -> int:
        """Get the number of patches that will be selected."""
        return max(1, int(self.num_patches * self.patch_percentage))

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the model configuration."""
        return {
            'model_name': self.__class__.__name__,
            'vit_backbone': self.vit_model_name,
            'color_img_size': self.color_img_size,
            'color_patch_size': self.color_patch_size,
            'ld_img_size': self.ld_img_size,
            'ld_patch_size': self.ld_patch_size,
            'num_patches': self.num_patches,
            'selected_patches': self.get_num_selected_patches(),
            'patch_percentage': self.patch_percentage,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
