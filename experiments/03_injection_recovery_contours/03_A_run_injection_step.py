import os
import torch
from blase.emulator import SparsePhoenixEmulator
from gollum.phoenix import PHOENIXSpectrum
import numpy as np
from muler.hpf import HPFSpectrum
import copy

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Pre-process the model as described in the paper
spectrum = PHOENIXSpectrum(teff=4100, logg=3.5)
spectrum = spectrum.divide_by_blackbody()
spectrum = spectrum.normalize()
continuum_fit = spectrum.fit_continuum(polyorder=5)
spectrum = spectrum.divide(continuum_fit, handle_meta="ff")

# Fetch the real HPF data
# For now we will use some data from another project
fn = "Goldilocks_20210919T084302_v1.0_0063.spectra.fits"
dir = "~/GitHub/star-witness/data/HPF/goldilocks/UT21-3-015/"
dir = os.path.expanduser(dir)
raw_data = HPFSpectrum(file=dir + fn, order=5)  # Just one order for now
data = raw_data.sky_subtract().deblaze().normalize().trim_edges((8, 2040))

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

state_dict_init = torch.load("emulator_T4100g3p5_prom0p01_HPF.pt")
emulator.load_state_dict(state_dict_init)

emulator.radial_velocity.requires_grad = False
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False

# ---------------------------------------------------------
# Injection step: Change ~20% of the amplitudes within a factor of ~2.7
# ---------------------------------------------------------
from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model.to(device)

n_lines = len(emulator.amplitudes)
perturbed_mask = torch.rand(n_lines) < 0.20
perturbed_mask = perturbed_mask.to(torch.float64)

offsets = 1.0 * (torch.rand(n_lines, dtype=torch.float64) - 0.9)
offsets_perturbed = offsets * perturbed_mask

frac_perturbed = perturbed_mask.sum() / n_lines

perturbed_state_dict = copy.deepcopy(emulator.state_dict())
fn_out = "emulator_T4100g3p5_prom0p01_HPF_injection.pt"

# Injected vsini and instrumental broadening.
fake_sigma_angs = torch.tensor(0.07)
fake_vsini = torch.tensor(4.1)

with torch.no_grad():
    rotationally_broadened = model.rotational_broaden(emulator.forward(), fake_vsini)
    instrumentally_broadened = model.instrumental_broaden(
        rotationally_broadened, fake_sigma_angs
    )
    noisefree_signal_perturbed = model.resample_to_data(instrumentally_broadened)

torch.save(noisefree_signal_perturbed, "noisefree_signal_perturbed.pt")


print("-" * 80)
print("Injecting a known perturbation into {:0.1%} of lines".format(frac_perturbed))
print("We are only perturbing amplitude in this experiment.")
print("Saving the file as: ", fn_out)
print("-" * 80)
perturbed_state_dict["amplitudes"] = perturbed_state_dict[
    "amplitudes"
] + offsets_perturbed.to(device)

torch.save(perturbed_state_dict, fn_out)
