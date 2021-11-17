import torch
from torch import nn
from tqdm import trange
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import numpy as np
import celerite2
from celerite2 import terms
import gpytorch
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def plot_spectrum(data):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch
    """
    fig, axes = plt.subplots(2, figsize=(8, 3))
    axes[0].step(
        data[0]["x"].numpy(), data[0]["y"].numpy(), label="Fit", lw=1,
    )
    axes[0].step(
        data[1]["x"].numpy(), data[1]["y"].numpy(), label="Input Data", lw=1,
    )
    axes[0].set_ylim(-10, 10)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].step(
        data[1]["x"].numpy(), data[0]["y"].numpy() - data[1]["y"].numpy(),
    )
    axes[1].set_ylim(-10, 10)
    return fig


# Set up parameterization
n_samples = 10000
x_vector = np.linspace(-1, 1, n_samples, endpoint=False)
true_sigma, true_rho = 1.7, 0.1
std_dev = 0.005
variance = std_dev ** 2
yerr = np.repeat(std_dev, n_samples)

# Set up GP with celerite
kernel = terms.SHOTerm(sigma=true_sigma, rho=true_rho, Q=0.25)
gp = celerite2.GaussianProcess(kernel, mean=0.0)
gp.compute(x_vector, yerr=None)

# Make fake signal
noise_free_signal = gp.sample()
noise_draw = np.random.normal(scale=yerr)
y_fake = noise_free_signal + noise_draw


writer = SummaryWriter(log_dir="runs/gpytorch1")
webbrowser.open("http://localhost:6006/", new=2)


## GPyTorch
train_x = torch.tensor(x_vector, dtype=torch.float64).to(device)
train_y = torch.tensor(y_fake, dtype=torch.float64).to(device)
train_yerr = torch.tensor(yerr, dtype=torch.float64).to(device)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=train_yerr ** 2
)  # GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

likelihood = likelihood.to(device)
model = model.to(device)
model.double()

## Training
model.train()
likelihood.train()

training_iterations = 500


# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1, amsgrad=True
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

t_iter = trange(training_iterations, desc="Training", leave=True)
for epoch in t_iter:
    # gpytorch.settings.skip_logdet_forward(state=False) as W:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))
    writer.add_scalar("loss", loss.item(), global_step=epoch)
    writer.add_scalar(
        "lengthscale",
        model.covar_module.base_kernel.lengthscale.item(),
        global_step=epoch,
    )
    writer.add_scalar(
        "amplitude", model.covar_module.outputscale.item(), global_step=epoch
    )
    # with torch.no_grad():
    #    model.eval()
    #    prediction = likelihood(model(train_x))
    #    to_plot = [
    #        {"x": train_x.cpu(), "y": train_y.cpu(),},
    #        {"x": train_x.cpu(), "y": prediction.mean.cpu()},
    #    ]
    #    writer.add_figure(
    #        "predictions vs. actuals", plot_spectrum(to_plot), global_step=epoch
    #    )


torch.save(model.state_dict(), "gpytorch_demo.pt")

