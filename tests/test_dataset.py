"""
Tests for dataset loading.
"""

import pytest
import torch
from pathlib import Path
from torchvision import transforms

from selective_magno_vit.data.dataset import ImageNetteDataset


class TestImageNetteDataset:
    """Tests for ImageNetteDataset."""

    @pytest.fixture
    def sample_transform(self):
        """Create a simple transform for testing."""
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def test_dataset_creation_fails_without_dirs(self, sample_transform):
        """Test that dataset creation fails when directories don't exist."""
        with pytest.raises(FileNotFoundError):
            ImageNetteDataset(
                magno_root="/nonexistent/magno",
                lines_root="/nonexistent/lines",
                split='train',
                transform=sample_transform
            )

    # Note: The following tests require actual data to be present
    # They are marked with @pytest.mark.integration to skip by default

    @pytest.mark.integration
    def test_dataset_loading(self, sample_transform):
        """Test loading dataset (requires actual data)."""
        # Update these paths to actual data locations
        magno_root = "data/preprocessed/magno_images"
        lines_root = "data/preprocessed/line_drawings"

        if not Path(magno_root).exists() or not Path(lines_root).exists():
            pytest.skip("Data directories not found")

        dataset = ImageNetteDataset(
            magno_root=magno_root,
            lines_root=lines_root,
            split='train',
            transform=sample_transform
        )

        assert len(dataset) > 0
        assert dataset.num_classes > 0

    @pytest.mark.integration
    def test_getitem(self, sample_transform):
        """Test getting a sample (requires actual data)."""
        magno_root = "data/preprocessed/magno_images"
        lines_root = "data/preprocessed/line_drawings"

        if not Path(magno_root).exists() or not Path(lines_root).exists():
            pytest.skip("Data directories not found")

        dataset = ImageNetteDataset(
            magno_root=magno_root,
            lines_root=lines_root,
            split='train',
            transform=sample_transform
        )

        sample = dataset[0]

        assert 'magno_image' in sample
        assert 'line_drawing' in sample
        assert 'label' in sample

        # Check shapes
        assert sample['magno_image'].shape[0] == 3  # RGB
        assert sample['line_drawing'].shape[0] == 1  # Grayscale
        assert sample['label'].dtype == torch.long

    @pytest.mark.integration
    def test_class_mapping(self, sample_transform):
        """Test class name to index mapping (requires actual data)."""
        magno_root = "data/preprocessed/magno_images"
        lines_root = "data/preprocessed/line_drawings"

        if not Path(magno_root).exists() or not Path(lines_root).exists():
            pytest.skip("Data directories not found")

        dataset = ImageNetteDataset(
            magno_root=magno_root,
            lines_root=lines_root,
            split='train',
            transform=sample_transform
        )

        # Check that class mappings are consistent
        for class_name, idx in dataset.class_to_idx.items():
            assert dataset.idx_to_class[idx] == class_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
