"""
Runtime helpers for generating and normalizing line drawings.
Uses the vendored informative-drawings Generator and keeps the API small/testable.
"""

from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF

if TYPE_CHECKING:
    import numpy as np

from selective_magno_vit.third_party.informative_drawings import Generator

TensorOrArray = Union[torch.Tensor, "np.ndarray"]  # type: ignore[name-defined]


def load_line_drawing_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    n_residual_blocks: int = 3,
    use_sigmoid: bool = True,
) -> torch.nn.Module:
    """
    Load an inference-ready Generator on the requested device.

    Args:
        checkpoint_path: Path to the pretrained Generator weights.
        device: torch.device to load the model onto.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Line drawing checkpoint not found at '{ckpt_path}'. "
            "Set data.line_drawings.checkpoint_path to a valid file."
        )

    model = Generator(
        input_nc=3,
        output_nc=1,
        n_residual_blocks=n_residual_blocks,
        sigmoid=use_sigmoid,
    )
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        # Handle common checkpoint wrappers.
        for key in ("state_dict", "model", "netG", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_aux_input(image: Union[Image.Image, torch.Tensor], out_size: int) -> torch.Tensor:
    """
    Build the Generator input tensor from a PIL image or tensor.
    The magno-style blur/grayscale is kept but naming remains neutral.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4 and image.size(0) == 1:
            image = image.squeeze(0)
        image = TF.to_pil_image(image)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image or torch.Tensor, got {type(image)}")

    # Convert to grayscale, apply a strong blur, then back to RGB (Generator expects 3 channels).
    aux_img = image.convert("L").filter(ImageFilter.GaussianBlur(radius=7.0)).convert("RGB")
    aux_img = aux_img.resize((out_size, out_size), Image.BICUBIC)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # match upstream test-time norm
        ]
    )
    aux_tensor = preprocess(aux_img).unsqueeze(0)  # (1,3,H,W)
    return aux_tensor


@torch.no_grad()
def infer_line_drawing(model: torch.nn.Module, aux_input: torch.Tensor) -> torch.Tensor:
    """Run the Generator to obtain a raw line drawing tensor."""
    device = next(model.parameters()).device
    inp = aux_input.to(device)
    return model(inp)


def normalize_line_drawing(
    raw: TensorOrArray,
    out_size: int,
    invert: bool = False,
    threshold: Optional[float] = None,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Normalize raw Generator output to float32 tensor in [0,1] with shape (1, out_size, out_size).
    Handles common GAN output ranges: [-1,1], [0,1], and [0,255].
    """
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore

    if isinstance(raw, torch.Tensor):
        tensor = raw.detach()
    elif np is not None and isinstance(raw, np.ndarray):  # type: ignore
        tensor = torch.from_numpy(raw)
    else:
        raise TypeError(f"Unsupported type for line drawing normalization: {type(raw)}")

    tensor = tensor.float()

    # Collapse batch/channel dimensions to a single-channel map.
    if tensor.dim() == 4:
        if tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        else:
            # Keep first sample; caller should feed per-image tensors.
            tensor = tensor[0]
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            pass
        else:
            tensor = tensor.mean(dim=0, keepdim=True)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(tensor.shape)}")

    # Rescale to [0,1].
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if min_val >= -1.1 and max_val <= 1.1 and min_val < 0:
        tensor = (tensor + 1.0) * 0.5
    elif min_val >= 0 and max_val > 1.5:
        tensor = tensor / 255.0
    # Else assume already in [0,1].

    if clamp:
        tensor = tensor.clamp(0.0, 1.0)

    if threshold is not None:
        tensor = (tensor >= threshold).float()

    if invert:
        tensor = 1.0 - tensor

    if tensor.shape[-2:] != (out_size, out_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if tensor.dim() != 3 or tensor.size(0) != 1:
        tensor = tensor.mean(dim=0, keepdim=True)

    return tensor.to(dtype=torch.float32)
