r"""
Emulator
--------------

Precomputed synthetic spectral models are awesome but imperfect and rigid.  Here we clone the most prominent spectral lines and continuum appearance of synthetic spectral models to turn them into tunable, flexible, semi-empirical models.  We can ultimately learn the properties of the pre-computed models with a neural network training loop, and then transfer those weights to real data, where a second transfer-learning training step can take place. The spectrum has :math:`N_{pix} \sim 300,000` pixels and :math:`N_{lines} \sim 5000` spectral lines.  The number of lines is set by the `prominence=` kwarg: lower produces more lines and higher (up to about 0.3) produces fewer lines.  


PhoenixEmulator
###############
"""
import math
import torch
from torch import nn
import os
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from astropy.io import fits

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
        # return self.sum_of_lorentzian_model(wl)
        return self.sum_of_voigts_model(wl)

    def sum_of_voigts_model(self, wl):
        """Return a sum-of-Voigts forward model, modulated by Blackbody and slopes"""
        net_spectrum = 1 - self.voigt_profile(
            self.lam_centers.unsqueeze(1),
            torch.exp(self.sigma_widths).unsqueeze(1),
            torch.exp(self.gamma_widths).unsqueeze(1),
            torch.exp(self.amplitudes).unsqueeze(1),
            wl.unsqueeze(0),
        ).sum(0)

        wl_normed = (wl - 10_500.0) / 2500.0
        modulation = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed ** 2
        )
        return net_spectrum * self.black_body(self.ln_teff_scalar, wl) * modulation

    def sum_of_lorentzian_model(self, wl):
        """Return the Lorentzian-only forward model, modulated by Blackbody and slopes"""
        net_spectrum = 1 - self.lorentzian_line(
            self.lam_centers.unsqueeze(1),
            torch.exp(self.sigma_widths).unsqueeze(1),
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
        peaks, _ = find_peaks(-flux_native, distance=10, prominence=prominence)
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

    def voigt_profile(
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
        y2 = y ** 2
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
