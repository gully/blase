import os
import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
from blase.emulator import SparsePhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
import numpy as np
from muler.hpf import HPFSpectrumList
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

state_dict_post = torch.load("emulator_T4700g4p5_prom0p01_HPF.pt")
emulator.load_state_dict(state_dict_post)


# Make synthetic data
# ---------------------------------------------------------
# Injection step: Change ~20% of the amplitudes within a factor of ~2.7
# ---------------------------------------------------------
from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model.to(device)

n_lines = len(emulator.amplitudes)
perturbed_mask = torch.rand(n_lines) < 0.20
perturbed_mask = perturbed_mask.to(torch.float64)

offsets = 1.0 * (torch.rand(n_lines, dtype=torch.float64) - 0.7)
offsets_perturbed = offsets * perturbed_mask

frac_perturbed = perturbed_mask.sum() / n_lines

perturbed_state_dict = copy.deepcopy(emulator.state_dict())
fn_out = "emulator_T4700g4p5_prom0p01_HPF_injection.pt"

# Injected vsini and instrumental broadening.
fake_sigma_angs = torch.tensor(0.07)
fake_vsini = torch.tensor(4.1)

print("-" * 80)
print("Injecting a known perturbation into {:0.1%} of lines".format(frac_perturbed))
print("We are only perturbing amplitude in this experiment.")
print("Saving the file as: ", fn_out)
print("-" * 80)
perturbed_state_dict["amplitudes"] = perturbed_state_dict[
    "amplitudes"
] + offsets_perturbed.to(device)

emulator.load_state_dict(perturbed_state_dict)

with torch.no_grad():
    rotationally_broadened = model.rotational_broaden(emulator.forward(), fake_vsini)
    instrumentally_broadened = model.instrumental_broaden(
        rotationally_broadened, fake_sigma_angs
    )
    noisefree_signal_perturbed = model.resample_to_data(instrumentally_broadened)

torch.save(noisefree_signal_perturbed, "noisefree_signal_perturbed.pt")
torch.save(perturbed_state_dict, fn_out)
