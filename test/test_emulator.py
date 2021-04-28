import torch
from blase.emulator import PhoenixEmulator


def test_forward():
    """Can we clone a model?"""

    emulator = PhoenixEmulator(4700, 4.5, prominence=0.1)
    wl = torch.arange(10500, 10600, 0.1)
    spectrum = emulator.forward(wl)

    assert spectrum is not None
    # len(spectrum) == 2048
    # assert spectrum.device.type == device
    # assert spectrum.dtype == torch.float64
