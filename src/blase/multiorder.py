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
import math
from torchinterp1d import Interp1d


class MultiOrder(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model 1D echelle spectra.

    Args:
        device (str): Either "cuda" for GPU acceleration, or "cpu" otherwise
        wl_limits (tuple): the limits :math:`\lambda_0` and :math:`\lambda_{max}` in Angstroms to analyze.
            Default: (425, 510)
    """

    def __init__(self, device="cuda", wl_data=None):
        super().__init__()

        self.device = device
        self.c_km_s = torch.tensor(2.99792458e5, device=device)
        self.n_pixels = 2048
        self.n_orders = 28
        self.pixel_index = torch.arange(self.n_pixels)
        self.root2pi = torch.sqrt(torch.tensor(2 * math.pi, device=device)).double()
        self.conv_window_size = 391  # must be odd for symmetry...
        self.conv_x = (
            torch.arange(self.conv_window_size, device=device).double()
            - self.conv_window_size // 2
        )

        # Need to init from data for wavelength resampling step
        if wl_data is None:
            raise Exception("Must provide data wavelenth grid to model")
        self.wl_data = wl_data.to(device)
        self.wl_0 = self.wl_data[0, 0]  # Hardcode for now
        self.wl_max = self.wl_data[-1, -1]

        # Set up a single echelle order
        wl_orig = fits.open(
            "/home/gully/libraries/raw/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        )[0].data.astype(np.float64)
        mask = (wl_orig > self.wl_0.item() * 0.995) & (
            wl_orig < self.wl_max.item() * 1.005
        )
        self.wl_native = torch.tensor(wl_orig[mask], device=device, dtype=torch.float64)

        flux_orig = fits.open(
            "/home/gully/libraries/raw/PHOENIX/Z-0.0/lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        )[0].data.astype(np.float64)
        self.flux_native = torch.tensor(
            flux_orig[mask], device=device, dtype=torch.float64
        )  # Units: erg/s/cm^2/cm

        self.native_median = torch.median(self.flux_native)
        self.flux_native /= self.native_median  # Units: Relative flux density

        self.v_z = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, dtype=torch.float64, device=device)
        )

        # self.log_blur_size = nn.Parameter(
        #    torch.tensor(-1.5, requires_grad=False, dtype=torch.float64, device=device)
        # )
        self.log_blur_size = torch.tensor(-1.5, dtype=torch.float64, device=device)

        xv = torch.linspace(-1, 1, 2048, device=device)

        self.cheb_coeffs = nn.Parameter(
            torch.tensor(
                self.n_orders * [[1.2, 0.1, -0.4, 0.15]],
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )
        )
        self.cheb_array = torch.stack(
            [
                torch.ones(self.n_pixels, device=device),
                xv,
                2 * xv ** 2 - 1,
                4 * xv ** 3 - 3 * xv,
            ]
        ).to(device)

    def forward(self, index):
        """The forward pass of the neural network model

        Args:
            index (int): the index of the ABB'A' nod frames: *e.g.* A=0, B=1, B'=2, A'=3
        Returns:
            (torch.tensor): the 2D generative scene model destined for backpropagation parameter tuning
        """

        # Radial Velocity Shift
        wl_0, wl_max = self.wl_data[index, 0] * 0.995, self.wl_data[index, -1] * 1.005
        rv_shift = torch.sqrt((self.c_km_s + self.v_z) / (self.c_km_s - self.v_z))
        wl_shifted = self.wl_native * rv_shift
        trim_mask = (wl_shifted > wl_0) & (wl_shifted < wl_max)

        pixel_scale = torch.mean(
            self.wl_native[trim_mask][1:] - self.wl_native[trim_mask][0:-1]
        )

        # Instrumental broadening
        blur_size_angstroms = torch.exp(self.log_blur_size)
        blur_size_pixels = blur_size_angstroms / pixel_scale
        normalization = torch.div(1.0, (blur_size_pixels * self.root2pi))

        weights = normalization * torch.exp(
            (-self.conv_x ** 2 / (2 * blur_size_pixels ** 2))
        )

        smoothed_flux = torch.nn.functional.conv1d(
            self.flux_native[trim_mask].unsqueeze(0).unsqueeze(1),
            weights.unsqueeze(0).unsqueeze(1),
            padding=self.conv_window_size // 2,
        ).squeeze()

        ## Resampling (This step is subtle to get right)
        ## match oversampled model to observed wavelengths
        ## For now simply interpolate.

        resampled_model_flux = Interp1d()(
            wl_shifted[trim_mask], smoothed_flux, self.wl_data[index]
        ).squeeze()

        # Blaze function warping
        blaze = (self.cheb_array * self.cheb_coeffs[index].unsqueeze(1)).sum(0)

        return resampled_model_flux * blaze
