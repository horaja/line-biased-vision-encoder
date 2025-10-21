"""
Tests for model components.
"""

import pytest
import torch

from selective_magno_vit.models.patch_scorer import PatchImportanceScorer
from selective_magno_vit.models.patch_selecter import SpatialThresholdSelector
from selective_magno_vit.models.selective_vit import SelectiveMagnoViT


class TestPatchImportanceScorer:
    """Tests for PatchImportanceScorer."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        patch_size = 4
        scorer = PatchImportanceScorer(patch_size=patch_size)

        # Create dummy line drawing
        batch_size = 2
        img_size = 64
        line_drawing = torch.rand(batch_size, 1, img_size, img_size)

        # Forward pass
        scores = scorer(line_drawing)

        # Check output shape
        num_patches = (img_size // patch_size) ** 2
        assert scores.shape == (batch_size, num_patches)

    def test_invalid_patch_size(self):
        """Test that invalid patch size raises error."""
        with pytest.raises(ValueError):
            PatchImportanceScorer(patch_size=0)

        with pytest.raises(ValueError):
            PatchImportanceScorer(patch_size=-1)

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        scorer = PatchImportanceScorer(patch_size=4)

        # 3D input (missing batch dimension)
        with pytest.raises(ValueError):
            scorer(torch.rand(1, 64, 64))

        # Wrong number of channels
        with pytest.raises(ValueError):
            scorer(torch.rand(2, 3, 64, 64))


class TestSpatialThresholdSelector:
    """Tests for SpatialThresholdSelector."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        batch_size = 2
        num_patches = 256
        embed_dim = 192
        img_size = 64

        selector = SpatialThresholdSelector(patch_percentage=0.5)

        # Create dummy inputs
        magno_patches = torch.rand(batch_size, num_patches, embed_dim)
        vit_pos_embed = torch.rand(1, num_patches + 1, embed_dim)  # +1 for CLS
        scores = torch.rand(batch_size, num_patches)
        line_drawing = torch.rand(batch_size, 1, img_size, img_size)

        # Forward pass
        selected_patches = selector(
            magno_patches,
            vit_pos_embed,
            scores,
            line_drawing
        )

        # Check output shape
        k = int(num_patches * 0.5)
        assert selected_patches.shape == (batch_size, k, embed_dim)

    def test_invalid_patch_percentage(self):
        """Test that invalid patch percentage raises error."""
        with pytest.raises(ValueError):
            SpatialThresholdSelector(patch_percentage=0)

        with pytest.raises(ValueError):
            SpatialThresholdSelector(patch_percentage=1.5)


class TestSelectiveMagnoViT:
    """Tests for SelectiveMagnoViT model."""

    @pytest.mark.slow
    def test_model_creation(self):
        """Test model can be created."""
        model = SelectiveMagnoViT(
            patch_percentage=0.4,
            num_classes=10,
            img_size=64,
            patch_size=4,
            pretrained=False  # Don't download weights for testing
        )
        assert model is not None

    @pytest.mark.slow
    def test_forward_pass(self):
        """Test full forward pass."""
        batch_size = 2
        img_size = 64
        num_classes = 10

        model = SelectiveMagnoViT(
            patch_percentage=0.4,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=4,
            pretrained=False
        )

        # Create dummy inputs
        magno_images = torch.rand(batch_size, 3, img_size, img_size)
        line_drawings = torch.rand(batch_size, 1, img_size, img_size)

        # Forward pass
        outputs = model(magno_images, line_drawings)

        # Check output shape
        assert outputs.shape == (batch_size, num_classes)

    @pytest.mark.slow
    def test_get_patch_indices(self):
        """Test getting selected patch indices."""
        model = SelectiveMagnoViT(
            patch_percentage=0.4,
            img_size=64,
            patch_size=4,
            pretrained=False
        )

        batch_size = 2
        line_drawings = torch.rand(batch_size, 1, 64, 64)

        indices = model.get_selected_patch_indices(line_drawings)

        num_patches = (64 // 4) ** 2
        k = int(num_patches * 0.4)
        assert indices.shape == (batch_size, k)

    @pytest.mark.slow
    def test_get_importance_map(self):
        """Test getting patch importance map."""
        img_size = 64
        patch_size = 4

        model = SelectiveMagnoViT(
            img_size=img_size,
            patch_size=patch_size,
            pretrained=False
        )

        batch_size = 2
        line_drawings = torch.rand(batch_size, 1, img_size, img_size)

        importance_map = model.get_patch_importance_map(line_drawings)

        expected_size = img_size // patch_size
        assert importance_map.shape == (batch_size, 1, expected_size, expected_size)

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        # Invalid patch percentage
        with pytest.raises(ValueError):
            SelectiveMagnoViT(patch_percentage=0, pretrained=False)

        # Image size not divisible by patch size
        with pytest.raises(ValueError):
            SelectiveMagnoViT(img_size=65, patch_size=4, pretrained=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
