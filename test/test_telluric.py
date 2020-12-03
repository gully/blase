import torch
from blase.telluric import TelluricModel
from blase.utils import suppress_stdout
import numpy as np
import astropy.units as u

with suppress_stdout():
    import hapi

    hapi.db_begin("../../hapi/data/")


def test_telluric_import():
    """Can we import the module?"""
    skymodel = TelluricModel()
    assert isinstance(skymodel, torch.nn.Module)


def test_telluric_forward():
    """Does the telluric forward pass work?"""
    skymodel = TelluricModel()
    output = skymodel.forward()
    assert isinstance(output, torch.Tensor)
    assert len(output) > 1000


def test_hitran_io():
    """Can we access and process HITRAN data"""
    skymodel = TelluricModel()

    wls = np.linspace(12600, 12800, 20000)
    nus = np.array((wls * u.Angstrom).to(1 / u.cm, equivalencies=u.spectral()).value)

    O2_hitran_all = skymodel.get_hapi_molec_data("O2")
    mask = (O2_hitran_all["nu"] > nus.min()) & (O2_hitran_all["nu"] < nus.max())

    assert mask.sum() < len(mask)
    assert mask.sum() > 10
