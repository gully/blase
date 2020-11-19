import torch
from blase.multiorder import MultiOrder


def test_forward():
    """Can we cast the model to GPU?"""

    device = "cpu"
    model = MultiOrder(device=device)
    spectrum = model.forward()

    assert len(spectrum) > 5
    assert spectrum.device.type == device
    assert spectrum.dtype == torch.float64
