import torch
from blase.multiorder import MultiOrder
from blase.datasets import HPFDataset


def test_forward():
    """Can we cast the model to GPU?"""

    device = "cpu"
    data = HPFDataset("data/Goldilocks_20191022T013208_v1.0_0003.spectra.fits")
    model = MultiOrder(device=device, wl_data=data.data_cube[6, :, :])
    spectrum = model.forward(5)

    assert len(spectrum) == 2048
    assert spectrum.device.type == device
    assert spectrum.dtype == torch.float64
