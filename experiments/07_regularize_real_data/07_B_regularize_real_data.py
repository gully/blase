import os
import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
from blase.emulator import SparsePhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
import numpy as np
from muler.hpf import HPFSpectrumList, HPFSpectrum
import copy
import astropy.units as u

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from astropy.io import fits

# Fetch the real HPF data
# We will demo on WASP 69
hdus = fits.open("../../data/WASP_69_hpf_stack.fits")

# Numpy arrays: 1 x N_pix
data = HPFSpectrum(
    flux=hdus[1].data["flux"] * u.dimensionless_unscaled,
    spectral_axis=hdus[1].data["wavelength"] * u.Angstrom,
)
data = data.remove_nans()


# Pre-process the model as described in the paper
spectrum = PHOENIXSpectrum(teff=4700, logg=4.5)
spectrum = spectrum.divide_by_blackbody()
spectrum = spectrum.normalize()
continuum_fit = spectrum.fit_continuum(polyorder=5)
spectrum = spectrum.divide(continuum_fit, handle_meta="ff")

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

emulator.radial_velocity = nn.Parameter(torch.tensor(-9.628, device=device))
emulator.radial_velocity.requires_grad = True
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False


from blase.emulator import EchelleModel

model = EchelleModel(
    data.spectral_axis.bin_edges.value.astype(np.float64), wl_native.cpu()
)
model.to(device)
model.ln_vsini = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

data_target = torch.tensor(
    data.flux.value.astype(np.float64), device=device, dtype=torch.float64
)

data_wavelength = torch.tensor(
    data.wavelength.value.astype(np.float64), device=device, dtype=torch.float64
)

loss_fn = nn.MSELoss(reduction="sum")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters()))
    + list(filter(lambda p: p.requires_grad, emulator.parameters())),
    0.01,
    amsgrad=True,
)
n_epochs = 1000
losses = []

with torch.no_grad():
    amplitude_init = copy.deepcopy(torch.exp(emulator.amplitudes))

# Define the prior on the amplitude
def ln_prior(amplitude_vector):
    """
    Prior for the amplitude vector
    """
    amplitude_difference = amplitude_init - torch.exp(amplitude_vector)
    return 0.5 * torch.sum((amplitude_difference ** 2) / (0.01 ** 2))


t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    model.train()
    emulator.train()
    high_res_model = emulator.forward()
    yhat = model.forward(high_res_model)
    loss = loss_fn(yhat, data_target)
    loss += ln_prior(emulator.amplitudes)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

torch.save(emulator.state_dict(), "emulator_T4700g4p5_prom0p01_HPF_MAP.pt")
torch.save(model.state_dict(), "extrinsic_MAP.pt")
