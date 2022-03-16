import torch
from blase.emulator import LinearEmulator, SparseLinearEmulator
from gollum.phoenix import PHOENIXSpectrum

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
