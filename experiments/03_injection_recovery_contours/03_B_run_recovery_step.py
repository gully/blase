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

state_dict_post = torch.load("emulator_T4100g3p5_prom0p01_HPF.pt")
emulator.load_state_dict(state_dict_post)

emulator.radial_velocity.requires_grad = False
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False

# ---------------------------------------------------------
# Synthetic Data training
# ---------------------------------------------------------
noisefree_signal = torch.load("noisefree_signal_perturbed.pt")

from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model_init = copy.deepcopy(model.state_dict())

# Do not train on real data.  Train on Synthetic data with known properties!
## data_target = torch.tensor(data.flux.value, device=device, dtype=torch.float64)

data_wavelength = torch.tensor(
    data.wavelength.value, device=device, dtype=torch.float64
)

loss_fn = nn.MSELoss(reduction="mean")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters()))
    + list(filter(lambda p: p.requires_grad, emulator.parameters())),
    0.05,
    amsgrad=True,
)
n_epochs = 1000
losses = []

# high_res_model = emulator.flux_native.clone().detach().to(device)

## We will save a dictionary-of-dictionaries:
dict_out1 = {}
dict_out2 = {}
n_draws = 5
for i in range(n_draws):
    # Draw a new noise vector each iteration
    noise_draw = np.random.normal(loc=0, scale=0.01, size=len(noisefree_signal))
    data_target = noisefree_signal.to(device) + torch.tensor(noise_draw).to(device)

    # Reset the models and optimizer back to their starting place.
    emulator.load_state_dict(state_dict_post)
    model.load_state_dict(model_init)

    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(filter(lambda p: p.requires_grad, emulator.parameters())),
        0.05,
        amsgrad=True,
    )

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

    with torch.no_grad():
        dict_out1[i] = copy.deepcopy(emulator.state_dict())
        dict_out2[i] = copy.deepcopy(model.state_dict())

torch.save(dict_out1, "emulator_T4100g3p5_prom0p01_HPF_recovery.pt")
torch.save(dict_out2, "model_recovery.pt")
