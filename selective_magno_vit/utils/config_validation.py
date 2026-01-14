"""
Lightweight configuration validation to catch line drawing/selection mismatches early.
"""

from pathlib import Path
from typing import Any


def validate_config(config: Any) -> None:
    """
    Validate critical configuration combinations to avoid silent misconfigurations.
    Raises ValueError/FileNotFoundError on invalid combinations.
    """
    mode = config.get("data.line_drawings.mode", "none")
    strategy = config.get("model.selector.strategy", "random")
    ld_checkpoint = config.get("data.line_drawings.checkpoint_path")
    ld_size = config.get("data.line_drawings.size", config.get("model.ld_img_size"))
    model_ld_size = config.get("model.ld_img_size")

    if mode not in ("none", "runtime"):
        raise ValueError(f"Unsupported data.line_drawings.mode '{mode}'. Use 'none' or 'runtime'.")

    if mode == "none" and strategy != "random":
        raise ValueError(
            "Line drawings are disabled (data.line_drawings.mode='none'), "
            "but a non-random selector strategy was requested. "
            "Enable runtime line drawings or set model.selector.strategy='random'."
        )

    if mode == "runtime":
        if not ld_checkpoint:
            raise ValueError(
                "Runtime line drawings enabled, but data.line_drawings.checkpoint_path is not set."
            )
        ckpt_path = Path(ld_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Line drawing checkpoint not found at '{ckpt_path}'. "
                "Set data.line_drawings.checkpoint_path to a valid file."
            )

    if ld_size is not None and model_ld_size is not None and ld_size != model_ld_size:
        raise ValueError(
            f"Configured line drawing size ({ld_size}) does not match model.ld_img_size ({model_ld_size}). "
            "Set data.line_drawings.size to the same value as model.ld_img_size."
        )
