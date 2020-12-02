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
        index, flux = data[5]
        assert isinstance(flux, torch.Tensor)
        assert len(flux) > 1900
