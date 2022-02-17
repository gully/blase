import os
import torch
from blase.emulator import SparsePhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
import numpy as np
from torch import nn
from muler.hpf import HPFSpectrum, HPFSpectrumList
import copy

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Pre-process the model as described in the paper
spectrum = PHOENIXSpectrum(teff=4700, logg=4.5)
spectrum = spectrum.divide_by_blackbody()
spectrum = spectrum.normalize()
continuum_fit = spectrum.fit_continuum(polyorder=5)
spectrum = spectrum.divide(continuum_fit, handle_meta="ff")

# Fetch the real HPF data
# For now we will use some data from another project
fn = "Goldilocks_20191022T014138_v1.0_0004.spectra.fits"
dir = "/home/gully/GitHub/blase/test/data/"
dir = os.path.expanduser(dir)
raw_data = HPFSpectrumList.read(file=dir + fn)  # Just one order for now
raw_data = raw_data.sky_subtract().deblaze().normalize().trim_edges((8, 2040))
raw_data = raw_data.stitch().mask_tellurics().normalize()

mask = (raw_data.wavelength.value > 8450) & (raw_data.wavelength.value < 8900)
data = raw_data.apply_boolean_mask(mask).normalize()


# Numpy arrays: 1 x N_pix
wl_native = spectrum.wavelength.value
flux_native = spectrum.flux.value

# Create the emulator and load a pretrained model
prominence = 0.01
emulator = SparsePhoenixEmulator(
    wl_native, flux_native, prominence=prominence, wing_cut_pixels=1000
)
emulator.to(device)

wl_native = emulator.wl_native.clone().detach().to(device)
wl_active = wl_native.to("cpu")[emulator.active_mask.to("cpu").numpy()]
target = (
    emulator.flux_native.clone().detach().to(device)[emulator.active_mask.cpu().numpy()]
)

state_dict_post = torch.load("emulator_T4700g4p5_prom0p01_HPF_MAP.pt")
emulator.load_state_dict(state_dict_post)

from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model.to(device)
data_target = torch.tensor(data.flux.value, device=device, dtype=torch.float64)

data_wavelength = torch.tensor(
    data.wavelength.value, device=device, dtype=torch.float64
)

model_state_dict = torch.load("extrinsic_MAP.pt")
model.load_state_dict(model_state_dict)

with torch.no_grad():
    processed_flux = model.forward(emulator.forward())

plt.step(data.wavelength.value, data.flux.value, label="Data")
plt.step(data.wavelength.value, processed_flux.cpu().numpy(), label="Model")
plt.legend()
plt.savefig("data_model_comparison_rigid_emulator.png", bbox_inches="tight", dpi=300)
