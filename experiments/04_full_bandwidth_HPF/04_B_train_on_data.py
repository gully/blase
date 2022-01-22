import os
import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
from blase.emulator import SparsePhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
import numpy as np
from muler.hpf import HPFSpectrum
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

state_dict_post = torch.load("emulator_T4700g4p5_prom0p01_HPF.pt")
emulator.load_state_dict(state_dict_post)

emulator.radial_velocity = nn.Parameter(torch.tensor(27.6, device=device))
emulator.radial_velocity.requires_grad = True
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False


from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model.ln_vsini = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

data_target = torch.tensor(data.flux.value, device=device, dtype=torch.float64)

data_wavelength = torch.tensor(
    data.wavelength.value, device=device, dtype=torch.float64
)

loss_fn = nn.MSELoss(reduction="mean")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters()))
    + list(filter(lambda p: p.requires_grad, emulator.parameters())),
    0.02,
    amsgrad=True,
)
n_epochs = 200
losses = []

t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    model.train()
    emulator.train()
    high_res_model = emulator.forward()
    yhat = model.forward(high_res_model)
    loss = loss_fn(yhat, data_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

torch.save(emulator.state_dict(), "emulator_T4700g4p5_prom0p01_HPF_MAP.pt")
torch.save(model.state_dict(), "extrinsic_MAP.pt")
