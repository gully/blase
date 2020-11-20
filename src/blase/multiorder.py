r"""
Spectral Model
--------------

We fit an echelle spectrum with :math:`M` total total orders, :math:`m`, with a single model. :math:`F(\lambda, \Theta)` where :math:`\Theta` are the set of stellar and nuisance parameters.


MultiOrder
##########
"""

import torch
from torch import nn
from astropy.io import fits
import numpy as np
import kornia


class MultiOrder(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model 1D echelle spectra.

    Args:
        device (str): Either "cuda" for GPU acceleration, or "cpu" otherwise
        wl_limits (tuple): the limits :math:`\lambda_0` and :math:`\lambda_{max}` in Angstroms to analyze.
            Default: (425, 510)
    """

    def __init__(self, device="cuda", init_from_data=None):
        super().__init__()

        self.device = device
        self.c_km_s = torch.tensor(2.99792458e5, device=device)

        # Need to init from data for wavelength resampling step
        if init_from_data is None:
            self.wl_data = torch.linspace(9773.25, 9899.2825, 2048, device=device)
        else:
            self.wl_data = init_from_data[6, 13, :]
        self.wl_0 = self.wl_data[0]  # Hardcode for now
        self.wl_max = self.wl_data[-1]

        # Set up a single echelle order
        wl_orig = fits.open(
            "/home/gully/libraries/raw/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        )[0].data.astype(np.float64)
        mask = (wl_orig > self.wl_0.item() * 0.995) & (
            wl_orig < self.wl_max.item() * 1.005
        )
        self.wl_native = torch.tensor(wl_orig[mask], device=device, dtype=torch.float64)

        flux_orig = fits.open(
            "/home/gully/libraries/raw/PHOENIX/Z-0.0/lte04500-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        )[0].data.astype(np.float64)
        self.flux_native = torch.tensor(
            flux_orig[mask], device=device, dtype=torch.float64
        )  # Units: erg/s/cm^2/cm

        self.native_median = torch.median(self.flux_native)
        self.flux_native /= self.native_median  # Units: Relative flux density

        self.scalar_const = nn.Parameter(
            torch.tensor(200.0, requires_grad=True, dtype=torch.float64, device=device)
        )

        self.v_z = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, dtype=torch.float64, device=device)
        )

        self.log_blur_size = nn.Parameter(
            torch.tensor(1.67, requires_grad=True, dtype=torch.float64, device=device)
        )

    def forward(self):
        """The forward pass of the neural network model

        Args:
            index (int): the index of the ABB'A' nod frames: *e.g.* A=0, B=1, B'=2, A'=3
        Returns:
            (torch.tensor): the 2D generative scene model destined for backpropagation parameter tuning
        """

        # Instrumental broadening
        blur_size = torch.exp(self.log_blur_size)
        smoothed_flux = kornia.filters.gaussian_blur2d(
            self.flux_native.view(1, 1, 1, -1),
            kernel_size=(1, 21),
            sigma=(0.01, blur_size),
        ).squeeze()

        # Radial Velocity Shift
        rv_shift = torch.sqrt((self.c_km_s + self.v_z) / (self.c_km_s - self.v_z))
        wl_shifted = self.wl_native * rv_shift
        trim_mask = (wl_shifted > self.wl_0) & (wl_shifted < self.wl_max)
        smoothed_flux = smoothed_flux[trim_mask]

        # Resampling (This step is subtle to get right)
        # match oversampled model to observed wavelengths:
        column_vector = self.wl_data.unsqueeze(1)
        row_vector = wl_shifted[trim_mask].unsqueeze(0)
        dist = (column_vector - row_vector) ** 2
        indices = dist.argmin(0)

        idx, vals = torch.unique(indices, return_counts=True)
        vs = torch.split_with_sizes(smoothed_flux, tuple(vals))
        resampled_model_flux = torch.tensor([v.mean() for v in vs])

        return resampled_model_flux * self.scalar_const
