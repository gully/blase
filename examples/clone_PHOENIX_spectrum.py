import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
from blase.emulator import PhoenixEmulator
import matplotlib.pyplot as plt
from gollum.phoenix import PHOENIXSpectrum
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import numpy as np

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


writer = SummaryWriter(log_dir="runs/emulator1")
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

# Create the emulator
emulator = PhoenixEmulator(wl_native, flux_native, prominence=0.02)
emulator.to(device)

n_pix = len(emulator.wl_native)
wl_native = emulator.wl_native.clone().detach().to(device)
target = emulator.flux_native.clone().detach().to(device)

# Training Loop
emulator.lam_centers.requires_grad = False
emulator.a_coeff.requires_grad = False
emulator.b_coeff.requires_grad = False
emulator.c_coeff.requires_grad = False

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, emulator.parameters()), 0.004, amsgrad=True
)
n_epochs = 100
sub_divisions = 50
losses = []

# Experiment: just optimize the line cores:
indices = np.isin(emulator.wl_native.cpu(), emulator.lam_centers.detach().cpu())
# from scipy.ndimage import binary_dilation
# dilated_inds = binary_dilation(indices, iterations=2)
peak_indices = torch.tensor(indices).to(device)

t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    for i in range(sub_divisions):
        emulator.train()
        if i in [0, 5, 10, 15, 20, 25, 30]:
            indices = peak_indices
        else:
            indices = torch.randint(0, n_pix, (n_pix // sub_divisions,)).to(device)
        wl = wl_native[indices].to(device)
        yhat = emulator.forward(wl)
        loss = loss_fn(yhat, target[indices])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

    writer.add_scalar("loss", loss.item(), global_step=epoch)
    writer.add_scalar("a", emulator.a_coeff.item(), global_step=epoch)
    writer.add_scalar("b", emulator.b_coeff.item(), global_step=epoch)
    writer.add_scalar("c", emulator.c_coeff.item(), global_step=epoch)
    if (epoch % 5) == 0:
        # torch.save(model.state_dict(), "model_coeffs.pt")
        sort_inds = np.argsort(wl.cpu())
        wl_plot = wl.cpu()[sort_inds]
        flux_clone = yhat.detach().cpu()[sort_inds]
        flux_targ = target[indices].cpu()[sort_inds]
        to_plot = [
            {"wl": wl_plot, "flux": flux_clone,},
            {"wl": wl_plot, "flux": flux_targ},
        ]
        writer.add_figure(
            "predictions vs. actuals", plot_spectrum(to_plot), global_step=epoch,
        )

torch.save(emulator.state_dict(), "native_res_0p1prom.pt")

