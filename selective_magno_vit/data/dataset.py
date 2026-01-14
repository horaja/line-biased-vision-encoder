"""
Dataset classes for SelectiveMagnoViT (Standard ImageNet-style support) with optional
runtime line drawing generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, get_worker_info
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from PIL import Image

from .line_drawings import (
    infer_line_drawing,
    load_line_drawing_model,
    make_aux_input,
    normalize_line_drawing,
)

logger = logging.getLogger(__name__)


class StandardImageDataset(datasets.ImageFolder):
    """
    Standard ImageFolder dataset wrapper for SelectiveMagnoViT.
    Can optionally generate line drawings at runtime.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Any],
        line_drawing_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.line_cfg = line_drawing_config or {}
        self.model_cfg = model_config or {}
        self.line_mode = self.line_cfg.get("mode", "none")
        self.ld_size = self.line_cfg.get("size", self.model_cfg.get("ld_img_size", 64))
        self.ld_threshold = self.line_cfg.get("threshold")
        self.ld_invert = self.line_cfg.get("invert", False)
        self.ld_clamp = self.line_cfg.get("clamp", True)
        self.ld_n_blocks = self.line_cfg.get("n_residual_blocks", 3)
        self.ld_device_str = self.line_cfg.get("device", "cpu")
        self.ld_checkpoint = self.line_cfg.get("checkpoint_path")

        super().__init__(root=root, transform=transform)
        self._worker_models: Dict[Optional[int], torch.nn.Module] = {}

    def _resolve_device(self) -> torch.device:
        if self.ld_device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_worker_model(self) -> torch.nn.Module:
        if self.line_mode != "runtime":
            return None  # type: ignore
        if not self.ld_checkpoint:
            raise ValueError("data.line_drawings.checkpoint_path must be set when mode='runtime'.")

        worker = get_worker_info()
        worker_id = worker.id if worker else None
        model = self._worker_models.get(worker_id)

        if model is None:
            device = self._resolve_device()
            model = load_line_drawing_model(
                checkpoint_path=self.ld_checkpoint,
                device=device,
                n_residual_blocks=self.ld_n_blocks,
            )
            self._worker_models[worker_id] = model
            logger.info(f"Loaded line drawing model on worker {worker_id} ({device})")

        return model

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, target = self.samples[index]
        image: Image.Image = self.loader(path).convert("RGB")

        color_tensor = self.transform(image) if self.transform is not None else image
        line_drawing = None

        if self.line_mode == "runtime":
            model = self._get_worker_model()
            aux = make_aux_input(image, self.ld_size)
            raw = infer_line_drawing(model, aux)
            line_drawing = normalize_line_drawing(
                raw,
                out_size=self.ld_size,
                invert=self.ld_invert,
                threshold=self.ld_threshold,
                clamp=self.ld_clamp,
            )

        return {
            "color_image": color_tensor,
            "line_drawing": line_drawing,
            "label": torch.tensor(target, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def collate_none_safe(batch):
    """
    Custom collate function that handles None values by returning None.
    """
    elem = batch[0]

    if isinstance(elem, dict):
        return {key: collate_none_safe([d[key] for d in batch]) for key in elem}
    if elem is None:
        return None
    return default_collate(batch)


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders from a standard ImageNet directory structure.
    
    Structure assumed:
        color_dir/
            train/
            val/
            
    Logic:
        - color_dir/val -> Held-out TEST set
        - color_dir/train -> Split into TRAIN and VALIDATION sets
    """
    img_size = config.get("model.color_img_size", 224)
    line_cfg = config.get("data.line_drawings", {}) or {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=config.get("training.augmentation.random_crop_scale", (0.08, 1.0))
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(**config.get("training.augmentation.color_jitter", {})),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),  # Resize slightly larger than target
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_root = Path(config.get("data.color_dir"))
    train_dir = data_root / "train"
    val_dir = data_root / "val"  # This becomes our held-out TEST set

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Could not find 'train' or 'val' directories in {data_root}")

    test_dataset = StandardImageDataset(
        root=str(val_dir),
        transform=eval_transform,
        line_drawing_config=line_cfg,
        model_config=config.get("model", {}),
    )

    full_train_dataset = StandardImageDataset(
        root=str(train_dir),
        transform=train_transform,
        line_drawing_config=line_cfg,
        model_config=config.get("model", {}),
    )

    full_train_for_val = StandardImageDataset(
        root=str(train_dir),
        transform=eval_transform,
        line_drawing_config=line_cfg,
        model_config=config.get("model", {}),
    )

    total_train_samples = len(full_train_dataset)
    train_size = int(0.9 * total_train_samples)
    val_size = total_train_samples - train_size

    generator = torch.Generator().manual_seed(config.get("experiment.seed", 42))
    train_subset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_subset = random_split(full_train_for_val, [train_size, val_size], generator=generator)

    num_workers = config.get("training.num_workers", 4)
    batch_size = config.get("training.batch_size", 32)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_none_safe,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_none_safe,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_none_safe,
    )

    logger.info(
        f"Dataset Split -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test (Held-out): {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
