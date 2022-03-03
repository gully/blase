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

emulator.radial_velocity = nn.Parameter(torch.tensor(0.0, device=device))
emulator.radial_velocity.requires_grad = False
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False


from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())
model.to(device)
model.ln_vsini = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

## The target is synthetic data for now
# data_target = torch.tensor(data.flux.value, device=device, dtype=torch.float64)
noisefree_signal = torch.load("noisefree_signal_perturbed.pt")
target = torch.tensor(
    np.random.normal(noisefree_signal.cpu().numpy(), 0.004), device=device
)

data_wavelength = torch.tensor(
    data.wavelength.value, device=device, dtype=torch.float64
)

loss_fn = nn.MSELoss(reduction="sum")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters()))
    + list(filter(lambda p: p.requires_grad, emulator.parameters())),
    0.01,
    amsgrad=True,
)
n_epochs = 2000
losses = []


def ln_prior(amplitude_vector):
    """
    Prior for the amplitude vector
    """
    return 0.5 * torch.sum((torch.exp(amplitude_vector) ** 2) / (0.1 ** 2))


# Assert fixed per-pixel uncertainty for now
per_pixel_uncertainty = torch.tensor(0.004, device=device, dtype=torch.float64)

t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    model.train()
    emulator.train()
    high_res_model = emulator.forward()
    yhat = model.forward(high_res_model)
    loss = loss_fn(yhat / per_pixel_uncertainty, target / per_pixel_uncertainty)
    loss += ln_prior(emulator.amplitudes)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

torch.save(emulator.state_dict(), "emulator_T4700g4p5_prom0p01_HPF_recovery.pt")
torch.save(model.state_dict(), "extrinsic_recovery.pt")
