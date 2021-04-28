r"""
Emulator
--------------

Precomputed synthetic spectral models are awesome but imperfect and rigid.  Here we clone the most prominent spectral lines and continuum appearance of synthetic spectral models to turn them into tunable, flexible, semi-empirical models.  We can ultimately learn the properties of the pre-computed models with a neural network training loop, and then transfer those weights to real data, where a second transfer-learning training step can take place. The spectrum has :math:`N_{pix} \sim 300,000` pixels and :math:`N_{lines} \sim 5000` spectral lines.  The number of lines is set by the `prominence=` kwarg: lower produces more lines and higher (up to about 0.3) produces fewer lines.  


PhoenixEmulator
###############
"""
import torch
from torch import nn
import os
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from astropy.io import fits


class PhoenixEmulator(nn.Module):
    r"""
    A PyTorch layer that clones precomputed synthetic spectra

    Teff (int): The Teff label of the PHOENIX model to read in.  Must be on the PHOENIX grid.
    logg (float): The logg label of the PHOENIX model to read in.  Must be on the PHOENIX grid.

    Currently hardcoded to assume your PHOENIX grid is stored at: ~/libraries/raw/PHOENIX/
    """

    def __init__(self, Teff, logg, prominence=0.03):
        super().__init__()

        self.Teff, self.logg = Teff, logg

        # Read in the synthetic spectra at native resolution
        self.wl_native, self.flux_native = self.read_native_PHOENIX_model(Teff, logg)

        (lam_centers, amplitudes, widths_angstroms,) = self.detect_lines(
            self.wl_native, self.flux_native, prominence=prominence
        )

        self.amplitudes = nn.Parameter(
            torch.log(amplitudes).clone().detach().requires_grad_(True)
        )
        self.widths = nn.Parameter(
            np.log(widths_angstroms).clone().detach().requires_grad_(True)
        )

        # Fix the wavelength centers as gospel for now.
        self.lam_centers = nn.Parameter(
            lam_centers.clone().detach().requires_grad_(False)
        )

        self.ln_teff_scalar = nn.Parameter(
            torch.tensor(0, requires_grad=True, dtype=torch.float64)
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

    def forward(self, wl):
        """The forward pass of the spectral model

        Returns:
            (torch.tensor): the 1D generative spectral model destined for backpropagation parameter tuning
        """

        net_spectrum = 1 - self.lorentzian_line(
            self.lam_centers.unsqueeze(1),
            torch.exp(self.widths).unsqueeze(1),
            torch.exp(self.amplitudes).unsqueeze(1),
            wl.unsqueeze(0),
        ).sum(0)

        wl_normed = (wl - 10_500.0) / 2500.0
        modulation = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed ** 2
        )

        return net_spectrum * self.black_body(self.ln_teff_scalar, wl) * modulation

    def black_body(self, ln_teff_scalar, wavelengths):
        """Make a black body spectrum given Teff and wavelengths
        
        Args:
            ln_teff_scalar (torch.tensor scalar): the natural log of a scalar multiplied by the baseline Teff
        Returns:
            (torch.tensor): the 1D smooth Black Body model normalized to roughly 1 for 4700 K
        -----
        """
        unnormalized = (
            1
            / (wavelengths / 10_000) ** 5
            / (
                torch.exp(
                    1.4387752e-2
                    / (wavelengths * 1e-10 * (self.Teff * torch.exp(ln_teff_scalar)))
                )
                - 1
            )
        )
        return unnormalized * 20.0

    def detect_lines(self, wl_native, flux_native, prominence=0.03):
        """Identify the spectral lines in the native model
        
        Args:
            wl_native (torch.tensor vector): The 1D vector of native model wavelengths (Angstroms)
            flux_native (torch.tensor vector): The 1D vector of native model fluxes (Normalized)
        Returns:
            (tuple of tensors): The wavelength centers, prominences, and widths for all ID'ed spectral lines
        -----
        """
        peaks, _ = find_peaks(-flux_native, distance=10, prominence=0.03)
        prominence_data = peak_prominences(-flux_native, peaks)
        width_data = peak_widths(-flux_native, peaks, prominence_data=prominence_data)
        lam_centers = wl_native[peaks]
        prominences = torch.tensor(prominence_data[0])
        widths = width_data[0]
        d_lam = np.diff(wl_native)[peaks]
        # Convert FWHM in pixels to Gaussian sigma in Angstroms
        widths_angs = torch.tensor(widths * d_lam / 2.355)

        return (lam_centers, prominences, widths_angs)

    def lorentzian_line(self, lam_center, width, amplitude, wavelengths):
        """Return a Lorentzian line, given properties"""
        return (
            amplitude
            / 3.141592654
            * width
            / (width ** 2 + (wavelengths - lam_center) ** 2)
        )

    def read_native_PHOENIX_model(
        self,
        teff,
        logg,
        PHOENIX_path="~/libraries/raw/PHOENIX/",
        wl_lo=8038,
        wl_hi=12849,
    ):
        """Return the native model wavelength and flux as a torch tensor
        
        Args:
            Teff (int): The Teff label of the PHOENIX model to read in.  Must be on the PHOENIX grid.
            logg (float): The logg label of the PHOENIX model to read in.  Must be on the PHOENIX grid.
            PHOENIX_path (str): The path to your local PHOENIX grid library.  You must have the PHOENIX
                grid downloaded locally.  Default: "~/libraries/raw/PHOENIX/"
            wl_lo (float): the bluest wavelength of the models to keep (Angstroms)
            wl_lo (float): the reddest wavelength of the models to keep (Angstroms)
        Returns:
            (tuple of tensors): the PHOENIX model wavelength and normalized flux at native spectral resolution
        """
        base_path = os.path.expanduser(PHOENIX_path)
        assert os.path.exists(
            base_path
        ), "You must specify the path to local PHOENIX models"

        wl_filename = base_path + "/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        assert os.path.exists(
            wl_filename
        ), "You need to place the PHOENIX models in {}".format(base_path)

        wl_orig = fits.open(wl_filename)[0].data.astype(np.float64)

        mask = (wl_orig > wl_lo) & (wl_orig < wl_hi)
        wl_out = torch.tensor(wl_orig[mask], dtype=torch.float64)

        fn = (
            base_path
            + "/Z-0.0/lte{:05d}-{:0.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        ).format(teff, logg)
        assert os.path.exists(fn), "Double check that the file {} exists".format(fn)

        flux_orig = fits.open(fn)[0].data.astype(np.float64)
        # Units: erg/s/cm^2/cm
        flux_native = torch.tensor(flux_orig[mask], dtype=torch.float64)
        native_median = torch.median(flux_native)
        # Units: Relative flux density
        flux_out = flux_native / native_median
        return (wl_out, flux_out)

