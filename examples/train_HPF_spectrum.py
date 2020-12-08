from os import initgroups

# from astropy.io.fits import column
# from numpy.lib.shape_base import column_stack
import torch
import time
from blase.datasets import HPFDataset
from blase.multiorder import MultiOrder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import webbrowser
import numpy as np

parser = argparse.ArgumentParser(
    description="Blase: astronomical echelle spectrum analysis with PyTorch"
)
parser.add_argument("filename", type=str, help="Goldilocks filename")
parser.add_argument(
    "--resume", action="store_true", help="Resume model from last existing saved model",
)
parser.add_argument(
    "--n_epochs", default=1800, type=int, help="Number of training epochs"
)

args = parser.parse_args()
print(args)

writer = SummaryWriter(log_dir="runs/exp1")
webbrowser.open("http://localhost:6006/", new=2)


def plot_spectrum(spectra):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch
    """
    fig, axes = plt.subplots(2, figsize=(8, 3))
    axes[0].step(
        spectra[0]["wl"].numpy(), spectra[0]["flux"].numpy(),
    )
    axes[0].step(
        spectra[1]["wl"].numpy(), spectra[1]["flux"].numpy(),
    )
    axes[1].step(
        spectra[1]["wl"].numpy(),
        spectra[0]["flux"].numpy() - spectra[1]["flux"].numpy(),
    )
    return fig


# Change to 'cpu' if you do not have an NVIDIA GPU
# Warning, it will be about 30X slower.
device = "cuda"

dataset = HPFDataset(args.filename)
data_cube = dataset.data_cube.to(device)
model = MultiOrder(device=device, wl_data=data_cube[6, :, :])
model = model.to(device, non_blocking=True)


# Initialize from a previous training run
if args.resume:
    # for key in model.state_dict():
    #    model.state_dict()[key] *=0
    #    model.state_dict()[key] += state_dict[key].to(device)
    state_dict = torch.load("model_coeffs.pt")
    model.load_state_dict(state_dict)

# Only send one frame per batch
n_frames_per_batch = 1
train_loader = DataLoader(dataset=dataset, batch_size=n_frames_per_batch, shuffle=True)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), 0.02)

# It currently takes 0.5 seconds per training epoch, for about 7200 epochs per hour
n_epochs = args.n_epochs

# Hard-code the 13th echelle order for now...
losses = []

columns = torch.arange(2048)
hard_mask = (columns > 10) & (columns < 2038)

t0 = time.time()
t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    for i, data in enumerate(train_loader, 0):
        index, spectrum = data
        y_batch = spectrum.squeeze().to(device)
        model.train()
        yhat = model.forward(index.item()).squeeze()
        loss = loss_fn(yhat[hard_mask], y_batch[hard_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    writer.add_scalar("loss", loss.item(), global_step=epoch)
    writer.add_scalar("RV", model.v_z.item(), global_step=epoch)
    writer.add_scalar("log blur size", model.log_blur_size.item(), global_step=epoch)
    t_iter.set_description(f"Loss {loss.item(): 15.3f}")
    t_iter.refresh()
    if (epoch % 20) == 0:
        torch.save(model.state_dict(), "model_coeffs.pt")
        to_plot = [
            {"wl": data_cube[6, index.item(), :].cpu(), "flux": yhat.detach().cpu()},
            {"wl": data_cube[6, index.item(), :].cpu(), "flux": y_batch.cpu()},
        ]
        writer.add_figure(
            "predictions vs. actuals", plot_spectrum(to_plot), global_step=epoch,
        )

# Save the model parameters for next time
t1 = time.time()
net_time = t1 - t0
print(f"{n_epochs} epochs on {device}: {net_time:0.1f} seconds", end="\t")
torch.save(model.state_dict(), "model_coeffs.pt")
writer.close()
