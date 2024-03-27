r"""
FadeevaEmulator
###############
"""
import math
import torch
from torch import nn
import numpy as np

from torch.special import erfc


def erfcx_naive(x):
    """Erfcx based on erfc"""
    return torch.exp(x * x) * erfc(x)


try:
    from torch.special import erfcx

    print("Woohoo! You have a version {} of PyTorch".format(torch.__version__))
except ImportError:
    erfcx = erfcx_naive
    print(
        "Version {} of PyTorch does not offer erfcx, defaulting to unstable...".format(
            torch.__version__
        )
    )


class FadeevaEmulator(nn.Module):
    r"""
    A PyTorch layer that clones precomputed synthetic spectra

    """

    def __init__(self, lam_centers, amplitudes, widths_angstroms):
        super().__init__()

        self.amplitudes = nn.Parameter(
            torch.log(amplitudes).clone().detach().requires_grad_(True)
        )
        self.sigma_widths = nn.Parameter(
            np.log(widths_angstroms / math.sqrt(2))
            .clone()
            .detach()
            .requires_grad_(True)
        )

        self.gamma_widths = nn.Parameter(
            np.log(widths_angstroms / math.sqrt(2))
            .clone()
            .detach()
            .requires_grad_(True)
        )

        # Fix the wavelength centers as gospel for now.
        self.lam_centers = nn.Parameter(
            lam_centers.clone().detach().requires_grad_(False)
        )

        self.a_coeff = nn.Parameter(
            torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
        )
        self.b_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
        )
        self.c_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
        )

        # Somehow these don't get told to go to cuda device with model.to(device)?
        # Send them to cuda manually as a HACK
        # TODO: figure out why .to(device) doesn't work on these attributes.
        self.an = (
            torch.tensor(
                [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    9.0,
                    9.5,
                    10.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                ]
            )
            .unsqueeze(0)
            .unsqueeze(1)
        ).cuda()

        self.a2n2 = (
            torch.tensor(
                [
                    0.25,
                    1.0,
                    2.25,
                    4.0,
                    6.25,
                    9.0,
                    12.25,
                    16.0,
                    20.25,
                    25.0,
                    30.25,
                    36.0,
                    42.25,
                    49.0,
                    56.25,
                    64.0,
                    72.25,
                    81.0,
                    90.25,
                    100.0,
                    110.25,
                    121.0,
                    132.25,
                    144.0,
                    156.25,
                    169.0,
                    182.25,
                ]
            )
            .unsqueeze(0)
            .unsqueeze(1)
        ).cuda()

    def forward(self, wl):
        """The forward pass of the spectral model

        Returns:
            (torch.tensor): the 1D generative spectral model destined for backpropagation parameter tuning
        """
        return self.product_of_voigts_model(wl)

    def product_of_voigts_model(self, wl):
        """Return a sum-of-Voigts forward model, modulated by Blackbody and slopes"""
        net_spectrum = (
            1
            - self.voigt_profile(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                torch.exp(self.gamma_widths).unsqueeze(1),
                torch.exp(self.amplitudes).unsqueeze(1),
                wl.unsqueeze(0),
            )
        ).prod(0)

        wl_normed = (wl - 10_500.0) / 2500.0
        modulation = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed**2
        )
        return net_spectrum * modulation

    def exact_voigt_profile(
        self, lam_center, sigma_width, gamma_width, amplitude, wavelengths
    ):
        """Return a tensor of sparse Voigt Profiles, given properties"""
        # At first the x term should be (N_lines x N_wl)
        x_term = (wavelengths - lam_center) / (math.sqrt(2) * sigma_width)
        # At first the a term should be (N_lines x 1)
        a_term = gamma_width / (math.sqrt(2) * sigma_width)
        prefactor = amplitude / (math.sqrt(2.0 * math.pi) * sigma_width)
        # xterm gains an empty dimension for approximation (N_lines x N_wl x 1)
        # aterm gains an empty dimension for approximation (N_lines x 1 x 1)
        unnormalized_voigt = self.rewofz(x_term.unsqueeze(2), a_term.unsqueeze(2))
        return prefactor * unnormalized_voigt.squeeze()

    def rewofz(self, x, y):
        """Real part of wofz (Faddeeva) function based on Algorithm 916

        We apply a=0.5 for Algorithm 916.
        Ported from exojax to PyTorch by gully

        Args:
            x: Torch tensor
        Must be shape (N_lines x N_wl x  1)
            y: Torch tensor
        Must be shape (N_lines x 1 x 1)

        Returns:
             Torch tensor:
             (N_wl x N_lines)
        """
        xy = x * y
        exx = torch.exp(-1.0 * x * x)
        f = exx * (
            erfcx(y) * torch.cos(2.0 * xy)
            + x * torch.sin(xy) / 3.141592654 * torch.sinc(xy / 3.141592654)
        )
        y2 = y**2
        Sigma23 = torch.sum(
            (torch.exp(-((self.an + x) ** 2)) + torch.exp(-((self.an - x) ** 2)))
            / (self.a2n2 + y2),
            axis=2,
        ).unsqueeze(2)

        Sigma1 = exx * (
            7.78800786e-01 / (0.25 + y2)
            + 3.67879450e-01 / (1.0 + y2)
            + 1.05399221e-01 / (2.25 + y2)
            + 1.83156393e-02 / (4.0 + y2)
            + 1.93045416e-03 / (6.25 + y2)
            + 1.23409802e-04 / (9.0 + y2)
            + 4.78511765e-06 / (12.25 + y2)
            + 1.12535176e-07 / (16.0 + y2)
        )

        f = f + y / math.pi * (-1 * torch.cos(2.0 * xy) * Sigma1 + 0.5 * Sigma23)
        return f
