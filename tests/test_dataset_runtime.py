from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from selective_magno_vit.data.dataset import StandardImageDataset


def test_runtime_line_drawing_generation(monkeypatch, tmp_path):
    # Create minimal ImageFolder structure
    root = tmp_path / "dataset" / "train" / "cls"
    root.mkdir(parents=True)
    img_path = root / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

    # Fake model + inference to avoid real checkpoints
    monkeypatch.setattr(
        "selective_magno_vit.data.dataset.load_line_drawing_model",
        lambda checkpoint_path, device, n_residual_blocks=3: object(),
    )
    monkeypatch.setattr(
        "selective_magno_vit.data.dataset.infer_line_drawing",
        lambda model, aux_input: torch.full((1, 1, 4, 4), 0.5),
    )

    line_cfg = {
        "mode": "runtime",
        "checkpoint_path": "dummy.ckpt",
        "size": 4,
        "invert": False,
    }
    model_cfg = {"ld_img_size": 4}

    dataset = StandardImageDataset(
        root=str(tmp_path / "dataset" / "train"),
        transform=transforms.ToTensor(),
        line_drawing_config=line_cfg,
        model_config=model_cfg,
    )

    sample = dataset[0]
    assert sample["line_drawing"] is not None
    assert sample["line_drawing"].shape == (1, 4, 4)
    assert torch.allclose(sample["line_drawing"], torch.full((1, 4, 4), 0.5))

    # Model should be cached per worker key (None for single-process)
    assert dataset._worker_models  # pylint: disable=protected-access
