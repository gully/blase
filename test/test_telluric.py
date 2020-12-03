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


def test_atomic_data_broadcasting():
    """Does the pytorch broadcasting make sense?"""

    skymodel = TelluricModel()
    molec_data = skymodel.get_hapi_molec_data("O2")

    wls = np.linspace(12600, 12800, 20000)
    nus = torch.tensor(
        (wls * u.Angstrom).to(1 / u.cm, equivalencies=u.spectral()).value,
        device=skymodel.device,
    )
    mask = (molec_data["nu"] > nus.min()) & (molec_data["nu"] < nus.max())
    for key in molec_data.keys():
        molec_data[key] = molec_data[key][mask]

    ## Computing the gamma HWHM
    gamma = skymodel.gamma_of_p_and_T(
        0.98,
        297.0,
        0.21,
        molec_data["n_air"],
        molec_data["gamma_air"],
        molec_data["gamma_self"],
    )

    print(f"\n\tFound {gamma.nelement()} O2 lines in local HITRAN")
    assert gamma.ndim == 1, "Gammas should usually be vectors"
    assert gamma.shape == molec_data["nu"].shape
    assert gamma.nelement() > 1

    ## Computing the Lorentz profile

    profiles = skymodel.lorentz_profile(
        nus.unsqueeze(1), 1.1, molec_data["nu"], gamma, molec_data["delta_air"], 1.0,
    )
    profile = profiles.sum(1)

    print(f"\n\tLorentz profile is {profiles.size()[0]} x {profiles.size()[1]}")
    assert profiles.ndim == 2
    assert profile.ndim == 1
    assert profile.nelement() == len(nus)

    q_value = skymodel.tips_Q_of_T(297.0, molec_data["gpp"], molec_data["elower"])

    assert q_value == q_value
    assert q_value.ndim == 0
    assert q_value.nelement() == 1

    S_ij = skymodel.S_ij_of_T(
        297.0,
        molec_data["sw"],
        molec_data["nu"],
        molec_data["gpp"],
        molec_data["elower"],
    )

    assert len(S_ij) == len(gamma)
    assert len(S_ij) == (S_ij == S_ij).sum()

