r"""
Spectral Model
-------

We fit an echelle spectrum with :math:`M` total total orders, :math:`m`, with a single model. :math:`F(\lambda, \Theta)` where :math:`\Theta` are the set of stellar and nuisance parameters. 


MultiOrder
############
"""

import torch
from torch import nn
from astropy.io import fits
import numpy as np


class MultiOrder(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model 1D echelle spectra.

    Args:
        device (str): Either "cuda" for GPU acceleration, or "cpu" otherwise
        wl_limits (tuple): the limits :math:`\lambda_0` and :math:`\lambda_{max}` in Angstroms to analyze.
            Default: (425, 510)
    """

    def __init__(self, device="cuda", wl_limits=(8_500, 12_0000), library_path=None):
        super().__init__()

        self.device = device
        self.wl_0 = wl_limits[0]
        self.wl_max = wl_limits[1]

        wl_orig = fits.open(
            "/home/gully/libraries/raw/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        )[0].data.astype(np.float64)
        mask = (wl_orig > self.wl_0) & (wl_orig < self.wl_max)
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

    def forward(self):
        """The forward pass of the neural network model

        Args:
            index (int): the index of the ABB'A' nod frames: *e.g.* A=0, B=1, B'=2, A'=3
        Returns:
            (torch.tensor): the 2D generative scene model destined for backpropagation parameter tuning
        """
        return self.flux_native * self.scalar_const
