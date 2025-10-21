"""
Tests for preprocessing pipeline.
"""

import pytest
from pathlib import Path

from selective_magno_vit.data.preprocessing import InformativeDrawingsPreprocessor


class TestInformativeDrawingsPreprocessor:
    """Tests for InformativeDrawingsPreprocessor."""

    def test_preprocessor_creation_fails_without_repo(self):
        """Test that preprocessor creation fails when repo doesn't exist."""
        with pytest.raises(FileNotFoundError):
            InformativeDrawingsPreprocessor(
                informative_drawings_path="/nonexistent/path"
            )

    @pytest.mark.integration
    def test_preprocessor_creation(self):
        """Test creating preprocessor (requires third_party repo)."""
        repo_path = "third_party/informative-drawings"

        if not Path(repo_path).exists():
            pytest.skip("Informative-drawings repository not found")

        preprocessor = InformativeDrawingsPreprocessor(
            informative_drawings_path=repo_path
        )

        assert preprocessor.repo_path.exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_dataset(self, tmp_path):
        """Test processing a small dataset (requires repo and GPU)."""
        repo_path = "third_party/informative-drawings"

        if not Path(repo_path).exists():
            pytest.skip("Informative-drawings repository not found")

        # Create temporary directories
        input_dir = tmp_path / "input"
        output_magno = tmp_path / "magno"
        output_lines = tmp_path / "lines"
        output_color = tmp_path / "color"

        input_dir.mkdir()

        # This test would need actual input images
        # Skipping actual processing test as it requires GPU and sample images
        pytest.skip("Requires GPU and sample images")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
