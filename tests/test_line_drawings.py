import torch

from selective_magno_vit.data.line_drawings import normalize_line_drawing


def _check_range_and_shape(tensor: torch.Tensor, out_size: int):
    assert tensor.dtype == torch.float32
    assert tensor.shape == (1, out_size, out_size)
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)


def test_normalize_neg1_to1_range():
    raw = torch.tensor([[-1.0, 0.0], [0.5, 1.0]])
    out = normalize_line_drawing(raw, out_size=2)
    _check_range_and_shape(out, 2)
    assert torch.isclose(out[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(out[0, 1, 1], torch.tensor(1.0))


def test_normalize_zero_to_one_passthrough():
    raw = torch.tensor([[0.2, 0.7], [0.5, 0.9]])
    out = normalize_line_drawing(raw, out_size=2)
    _check_range_and_shape(out, 2)
    assert torch.isclose(out[0, 0, 0], raw[0, 0])


def test_normalize_uint8_like():
    raw = torch.full((2, 2), 255.0)
    out = normalize_line_drawing(raw, out_size=2)
    _check_range_and_shape(out, 2)
    assert torch.isclose(out.max(), torch.tensor(1.0))


def test_normalize_channels_and_resize():
    raw = torch.rand(3, 2, 2)  # multi-channel collapses to one
    out = normalize_line_drawing(raw, out_size=4)
    _check_range_and_shape(out, 4)


def test_normalize_threshold_and_invert():
    raw = torch.tensor([[0.2, 0.8], [0.5, 0.4]])
    out = normalize_line_drawing(raw, out_size=2, threshold=0.5, invert=True, clamp=True)
    _check_range_and_shape(out, 2)
    # After threshold + invert, 0.8 -> 0 -> invert ->1; 0.2 ->0 -> invert ->1
    assert torch.equal(out, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
