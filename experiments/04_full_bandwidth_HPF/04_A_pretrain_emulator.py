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

emulator.radial_velocity.requires_grad = False
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = True
emulator.gamma_widths.requires_grad = True

loss_fn = nn.MSELoss(reduction="mean")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, emulator.parameters())), 0.01, amsgrad=True,
)
n_epochs = 1000
losses = []

# high_res_model = emulator.flux_native.clone().detach().to(device)
t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    emulator.train()
    yhat = emulator.forward()[emulator.active_mask]
    loss = loss_fn(yhat, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

torch.save(emulator.state_dict(), "emulator_T4700g4p5_prom0p01_HPF.pt")
