import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
import torch.profiler
from blase.emulator import SparsePhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import numpy as np
import pandas as pd

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


log_dir = "runs/sparse1"
writer = SummaryWriter(log_dir=log_dir)
webbrowser.open("http://localhost:6006/", new=2)


# Pre-process the model as described in the paper
spectrum = PHOENIXSpectrum(teff=4700, logg=4.5)
spectrum = spectrum.divide_by_blackbody()
spectrum = spectrum.normalize()
continuum_fit = spectrum.fit_continuum(polyorder=5)
spectrum = spectrum.divide(continuum_fit, handle_meta="ff")

# Numpy arrays: 1 x N_pix
wl_native = spectrum.wavelength.value
flux_native = spectrum.flux.value

# Vary the prominence parameter
prominences = [0.2, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001]
final_stddev = [0.0] * len(prominences)
total_nlines = [0.0] * len(prominences)
for i, prominence in enumerate(prominences):
    emulator = SparsePhoenixEmulator(
        wl_native, flux_native, prominence=prominence, device=None, wing_cut_pixels=100
    )
    emulator.to(device)
    target = (
        emulator.flux_native.clone()
        .detach()
        .to(device)[emulator.active_mask.cpu().numpy()]
    )

    n_lines = len(emulator.lam_centers.detach())

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, emulator.parameters()), 0.03, amsgrad=True
    )
    n_epochs = 200
    losses = []
    std_devs = []

    t_iter = trange(n_epochs, desc="Training", leave=True)
    for epoch in t_iter:
        emulator.train()
        yhat = emulator.forward()
        loss = loss_fn(yhat, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))
        with torch.no_grad():
            std_dev = torch.std(yhat.detach() - target)
            std_devs.append(std_dev)

        writer.add_scalar("loss", loss.item(), global_step=epoch + i * n_epochs)
        writer.add_scalar("stddev", std_dev.item(), global_step=epoch + i * n_epochs)
        writer.add_scalar("prom", prominence, global_step=epoch + i * n_epochs)
        writer.add_scalar("n_lines", n_lines, global_step=epoch + i * n_epochs)

    final_stddev[i] = std_dev.item()
    total_nlines[i] = n_lines
    filename_out = "sparse_T4700g4p5_prom0p{:03d}_HPF.pt".format(int(prominence * 1000))
    torch.save(emulator.state_dict(), filename_out)
    df = pd.DataFrame(
        {"prominences": prominences, "n_lines": total_nlines, "std_dev": final_stddev}
    )
    df.to_csv("accuracy_vs_Prom.csv", index=False)
