import torch
from blase.datasets import HPFDataset
import glob


def test_import():
    """Can we import the module?"""
    fns = glob.glob("data/Goldilocks_*.fits")
    for fn in fns:
        data = HPFDataset(fn)
        assert isinstance(data, torch.utils.data.Dataset)


def test_pixels():
    """Can we import the module?"""
    fns = glob.glob("data/Goldilocks_*.fits")
    for fn in fns:
        data = HPFDataset(fn)
        wl, flux, unc = data[0]
        assert isinstance(wl, torch.Tensor)
        assert len(flux) > 10_000
        assert flux.shape == unc.shape
        assert wl.shape == flux.shape
