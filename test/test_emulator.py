import torch
from blase.emulator import (
    LinearEmulator,
    SparseLinearEmulator,
    SparseLogEmulator,
    ExtrinsicModel,
    InstrumentalModel,
)
from gollum.phoenix import PHOENIXSpectrum
import numpy as np

spectrum = PHOENIXSpectrum(teff=4700, logg=4.5)
spectrum = spectrum.divide_by_blackbody()
spectrum = spectrum.normalize()
continuum_fit = spectrum.fit_continuum(polyorder=5)
spectrum = spectrum.divide(continuum_fit, handle_meta="ff")
spec_native = spectrum


def test_dense_forward():
    """Can we clone a model?"""

    wl_native = spec_native.wavelength.value
    flux_native = spec_native.flux.value

    emulator = LinearEmulator(wl_native, flux_native, prominence=0.02)
    wl = torch.arange(10500, 10600, 0.1)
    spectrum = emulator.forward(wl)

    assert spectrum is not None
    assert len(spectrum) == len(wl)
    assert spectrum.dtype == torch.float64


def test_sparse_forward():
    """Can we clone a model?"""

    wl_native = spec_native.wavelength.value
    flux_native = spec_native.flux.value

    emulator = SparseLinearEmulator(
        wl_native, flux_native, prominence=0.02, wing_cut_pixels=1000
    )
    spectrum = emulator.forward()

    assert spectrum is not None
    assert len(spectrum) > 0
    assert spectrum.dtype == torch.float64

    model = ExtrinsicModel(wl_native)
    out1 = model.forward(torch.tensor(flux_native))
    out2 = model.forward(spectrum)

    assert model is not None
    assert out1 is not None
    assert out2 is not None
    assert len(out1) == len(out2)

    bin_edges = spec_native.spectral_axis.bin_edges.value.astype(np.float64)
    inst_model = InstrumentalModel(bin_edges, wl_native)
    out1 = inst_model.forward(torch.tensor(flux_native))
    out2 = inst_model.forward(spectrum)

    assert inst_model is not None
    assert out1 is not None
    assert out2 is not None
    assert len(out1) == len(out2)

    # Test the LogEmulator

    lnflux_native = torch.log(torch.tensor(flux_native))

    log_emulator = SparseLogEmulator(
        wl_native, lnflux_native, prominence=0.02, wing_cut_pixels=1000
    )
    spectrum_flux = log_emulator.forward()

    assert spectrum_flux is not None
    assert len(spectrum_flux) > 0
    assert spectrum_flux.dtype == torch.float64
