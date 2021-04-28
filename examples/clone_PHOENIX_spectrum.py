import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
from blase.emulator import PhoenixEmulator

# Toggle below if you just want to read in a pre-trained model
redo_training = True
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Create the emulator
emulator = PhoenixEmulator(4700, 4.5, prominence=0.1)
emulator.to(device)

n_pix = len(emulator.wl_native)
wl_native = emulator.wl_native.clone().detach().to(device)
target = emulator.flux_native.clone().detach().to(device)

if redo_training:
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(emulator.parameters(), 0.004)
    n_epochs = 100
    sub_divisions = 20
    losses = []

    t_iter = trange(n_epochs, desc="Training", leave=True)
    for epoch in t_iter:
        for i in range(sub_divisions):
            emulator.train()
            indices = torch.randint(0, n_pix, (n_pix // sub_divisions,)).to(device)
            wl = wl_native[indices].to(device)
            yhat = emulator.forward(wl)
            loss = loss_fn(yhat, target[indices])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))

    torch.save(emulator.state_dict(), "native_res_0p1prom.pt")
else:
    emulator.load_state_dict(torch.load("native_res_0p1prom.pt", map_location=device))

