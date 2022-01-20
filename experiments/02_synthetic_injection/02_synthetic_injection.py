import os
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
from muler.hpf import HPFSpectrum

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def plot_spectrum(spectra):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch
    """
    fig, axes = plt.subplots(2, figsize=(8, 3))
    axes[0].step(
        spectra[0]["wl"].numpy(),
        spectra[0]["flux"].numpy(),
        label="Cloned Spectrum",
        lw=1,
    )
    axes[0].step(
        spectra[1]["wl"].numpy(),
        spectra[1]["flux"].numpy(),
        label="Native Spectrum",
        lw=1,
    )
    axes[0].set_ylim(0, 1.5)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].step(
        spectra[1]["wl"].numpy(),
        spectra[0]["flux"].numpy() - spectra[1]["flux"].numpy(),
    )
    axes[1].set_ylim(-0.3, 0.3)
    return fig


log_dir = "runs/synthetic1"
writer = SummaryWriter(log_dir=log_dir)
webbrowser.open("http://localhost:6006/", new=2)


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
    wl_native, flux_native, prominence=prominence, wing_cut_pixels=100
)
emulator.to(device)

wl_native = emulator.wl_native.clone().detach().to(device)
wl_active = wl_native.to("cpu")[emulator.active_mask.to("cpu").numpy()]
target = (
    emulator.flux_native.clone().detach().to(device)[emulator.active_mask.cpu().numpy()]
)

state_dict_post = torch.load("sparse_T4100g3p5_prom0p01_HPF.pt")
emulator.load_state_dict(state_dict_post)

emulator.radial_velocity.requires_grad = False
emulator.lam_centers.requires_grad = False
emulator.amplitudes.requires_grad = True
emulator.sigma_widths.requires_grad = False
emulator.gamma_widths.requires_grad = False

# ---------------------------------------------------------
# Synthetic Data training
# ---------------------------------------------------------
# Read in the fake data
synthetic_spectrum = torch.load("./synthetic_perturbed_spectrum.pt")

from blase.emulator import EchelleModel

model = EchelleModel(data.spectral_axis.bin_edges.value, wl_native.cpu())

# Do not train on real data.  Train on Synthetic data with known properties!
## data_target = torch.tensor(data.flux.value, device=device, dtype=torch.float64)
data_target = synthetic_spectrum.to(device)

data_wavelength = torch.tensor(
    data.wavelength.value, device=device, dtype=torch.float64
)

loss_fn = nn.MSELoss(reduction="mean")

optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters()))
    + list(filter(lambda p: p.requires_grad, emulator.parameters())),
    0.01,
    amsgrad=True,
)
n_epochs = 5000
losses = []

# high_res_model = emulator.flux_native.clone().detach().to(device)

plot_every_N_steps = 100
t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    model.train()
    emulator.train()
    high_res_model = emulator.forward()
    yhat = model.forward(high_res_model)
    loss = loss_fn(yhat, data_target)
    loss.backward()
    optimizer.step()
    # for name, param in model.named_parameters():
    #    print(name, param.grad)
    # for name, param in emulator.named_parameters():
    #    print(name, param.grad)
    optimizer.zero_grad()
    losses.append(loss.item())
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

    writer.add_scalar("loss", loss.item(), global_step=epoch)
    writer.add_scalar(
        "sigma_angs", 0.01 + np.exp(model.ln_sigma_angs.item()), global_step=epoch
    )
    writer.add_scalar("vsini", 0.9 + np.exp(model.ln_vsini.item()), global_step=epoch)
    writer.add_scalar("RV", emulator.radial_velocity.item(), global_step=epoch)
    if (epoch % plot_every_N_steps) == 0:
        # torch.save(model.state_dict(), "model_coeffs.pt")
        wl_plot = data_wavelength.cpu()
        flux_clone = yhat.detach().cpu()
        flux_targ = data_target.cpu()
        to_plot = [
            {"wl": wl_plot, "flux": flux_clone,},
            {"wl": wl_plot, "flux": flux_targ},
        ]
        writer.add_figure(
            "predictions vs. actuals", plot_spectrum(to_plot), global_step=epoch,
        )

torch.save(model.state_dict(), "model_T4100g3p5_prom0p01_HPF_recovery.pt")
