import torch
from blase.telluric import TelluricModel


def test_import():
    """Can we import the module?"""
    skymodel = TelluricModel()
    assert isinstance(skymodel, torch.nn.Module)


def test_forward():
    """Can we import the module?"""
    skymodel = TelluricModel()
    output = skymodel.forward()
    assert isinstance(output, torch.Tensor)
    assert len(output) > 1000

