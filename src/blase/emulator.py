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
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from tqdm import tqdm


class PhoenixEmulator(nn.Module):
    r"""
    A PyTorch layer that clones precomputed synthetic spectra

    wl_native (int): The input wavelength
    flux_native (float): The output wavelength

    Currently hardcoded to assume your PHOENIX grid is stored at: ~/libraries/raw/PHOENIX/
    """

    def __init__(self, wl_native, flux_native, prominence=0.03):
        super().__init__()

        # Read in the synthetic spectra at native resolution
        self.wl_native = torch.tensor(wl_native)
        self.flux_native = torch.tensor(flux_native)

        (
            lam_centers,
            amplitudes,
            widths_angstroms,
        ) = self.detect_lines(self.wl_native, self.flux_native, prominence=prominence)

        self.n_pix = len(wl_native)

        line_buffer = 30  # Angstroms
        self.wl_min = wl_native.min()
        self.wl_max = wl_native.max()

        ## Set up "active area", where the region-of-interest is:
        # active_buffer = 60  # Angstroms
        # active_lower, active_upper = (
        #    self.wl_min + active_buffer,
        #    self.wl_max - active_buffer,
        # )

        # Set up line threshold, where lines are computed outside the active area
        line_threshold_lower, line_threshold_upper = (
            self.wl_min + line_buffer,
            self.wl_max - line_buffer,
        )

        # Restrict the lines to the active region plus 30 A buffer region

        mask = (lam_centers > line_threshold_lower) & (
            lam_centers < line_threshold_upper
        )
        lam_centers = lam_centers[mask]
        amplitudes = amplitudes[mask]
        widths_angstroms = widths_angstroms[mask]
        self.n_lines = len(lam_centers)

        # Experimentally determined scale factor tweaks
        amp_tweak = 0.14
        sigma_width_tweak = 1.28
        gamma_width_tweak = 1.52

        self.amplitudes = nn.Parameter(
            torch.log(amplitudes * amp_tweak).clone().detach().requires_grad_(True)
        )
        self.sigma_widths = nn.Parameter(
            np.log(widths_angstroms / math.sqrt(2) * sigma_width_tweak)
            .clone()
            .detach()
            .requires_grad_(True)
        )

        self.gamma_widths = nn.Parameter(
            np.log(widths_angstroms / math.sqrt(2) * gamma_width_tweak)
            .clone()
            .detach()
            .requires_grad_(True)
        )

        # Fix the wavelength centers as gospel for now.
        self.lam_centers = nn.Parameter(
            lam_centers.clone().detach().requires_grad_(False)
        )

        self.a_coeff = nn.Parameter(
            torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        )
        self.b_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=False, dtype=torch.float64)
        )
        self.c_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=False, dtype=torch.float64)
        )

    def forward(self, wl):
        """The forward pass of the spectral model

        Returns:
            (torch.tensor): the 1D generative spectral model destined for backpropagation parameter tuning
        """
        # return self.product_of_lorentzian_model(wl)
        return self.product_of_pseudovoigt_model(wl)

    def product_of_lorentzian_model(self, wl):
        """Return the Lorentzian-only forward model, modulated by Blackbody and slopes"""
        net_spectrum = (
            1
            - self.lorentzian_line(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                torch.exp(self.amplitudes).unsqueeze(1),
                wl.unsqueeze(0),
            )
        ).prod(0)

        wl_normed = (wl - 10_500.0) / 2500.0
        modulation = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed ** 2
        )
        return net_spectrum * modulation

    def product_of_pseudovoigt_model(self, wl):
        """Return the PseudoVoight forward model"""
        net_spectrum = (1 - self.pseudo_voigt_profiles(wl)).prod(0)

        wl_normed = (wl - 10_500.0) / 2500.0
        modulation = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed ** 2
        )
        return net_spectrum * modulation

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

    def lorentzian_line(self, lam_center, width, wavelengths):
        """Return a Lorentzian line, given properties"""
        return 1 / 3.141592654 * width / (width ** 2 + (wavelengths - lam_center) ** 2)

    def gaussian_line(self, lam_center, width, wavelengths):
        """Return a normalized Gaussian line, given properties"""

        return (
            1.0
            / (width * 2.5066)
            * torch.exp(-0.5 * ((wavelengths - lam_center) / width) ** 2)
        )

    def _compute_eta(self, fwhm_L, fwhm):
        """Compute the eta parameter for pseudo Voigt"""
        f_ratio = fwhm_L / fwhm
        return 1.36603 * f_ratio - 0.47719 * f_ratio ** 2 + 0.11116 * f_ratio ** 3

    def _compute_fwhm(self, fwhm_L, fwhm_G):
        """Compute the fwhm for pseudo Voigt using the approximation:
        :math:`f = [f_G^5 + 2.69269 f_G^4 f_L + 2.42843 f_G^3 f_L^2 + 4.47163 f_G^2 f_L^3 + 0.07842 f_G f_L^4 + f_L^5]^{1/5}`

        """

        return (
            fwhm_G ** 5
            + 2.69269 * fwhm_G ** 4 * fwhm_L ** 1
            + 2.42843 * fwhm_G ** 3 * fwhm_L ** 2
            + 4.47163 * fwhm_G ** 2 * fwhm_L ** 3
            + 0.07842 * fwhm_G ** 1 * fwhm_L ** 4
            + fwhm_L ** 5
        ) ** (1 / 5)

    def pseudo_voigt_profiles(self, wavelengths):
        """Compute the pseudo Voigt Profile, much faster than the full Voigt profile"""
        fwhm_G = 2.3548 * torch.exp(self.sigma_widths).unsqueeze(1)
        fwhm_L = 2.0 * torch.exp(self.gamma_widths).unsqueeze(1)
        fwhm = self._compute_fwhm(fwhm_L, fwhm_G)
        eta = self._compute_eta(fwhm_L, fwhm)

        return torch.exp(self.amplitudes).unsqueeze(1) * (
            eta
            * self.lorentzian_line(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.gamma_widths).unsqueeze(1),
                wavelengths.unsqueeze(0),
            )
            + (1 - eta)
            * self.gaussian_line(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                wavelengths.unsqueeze(0),
            )
        )


class SparsePhoenixEmulator(PhoenixEmulator):
    r"""
    A sparse implementation of the PhoenixEmulator

    wl_native (float vector): The input wavelength
    flux_native (float vector): The native flux
    prominence (int scalar): The threshold for detecting lines
    """

    def __init__(self, wl_native, flux_native, prominence=0.03):
        super().__init__(wl_native, flux_native, prominence=prominence)

        ## Define the wing cut
        # Currently defined in *pixels*
        wing_cut_pixels = 6000

        with torch.no_grad():
            list_of_nonzero_indices = []
            for line_center in tqdm(self.lam_centers.detach().numpy()):
                distance = np.abs(wl_native - line_center)
                these_pixels = np.argsort(distance)[0:wing_cut_pixels]
                list_of_nonzero_indices.append(sorted(these_pixels))

            self.indices_2D = torch.tensor(list_of_nonzero_indices)
            self.indices_1D = self.indices_2D.reshape(-1)
            self.indices = self.indices_1D.unsqueeze(0)
            self.wl_2D = self.wl_native[self.indices_2D]
            self.wl_1D = self.wl_2D.reshape(-1)

    def forward(self):
        """The forward pass of the sparse implementation--- no wavelengths needed!

        Returns:
            (torch.tensor): the 1D generative spectral model destined for backpropagation parameter tuning
        """
        flux_2D = torch.exp(self.amplitudes).unsqueeze(1) * self.gaussian_line(
            self.lam_centers.unsqueeze(1),
            torch.exp(self.sigma_widths).unsqueeze(1),
            self.wl_2D,
        )

        flux_1D = flux_2D.reshape(-1)
        ln_term = torch.log(1 - flux_1D)

        sparse_matrix = torch.sparse_coo_tensor(
            self.indices, ln_term, size=(self.n_pix,), requires_grad=False
        )

        result_1D = sparse_matrix.to_dense()

        return torch.exp(result_1D)
