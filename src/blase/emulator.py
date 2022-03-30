r"""
Emulator
--------------

Precomputed synthetic spectral models are awesome but imperfect and rigid.  Here we clone the most prominent spectral lines and continuum appearance of synthetic spectral models to turn them into tunable, flexible, semi-empirical models.  We can ultimately learn the properties of the pre-computed models with a neural network training loop, and then transfer those weights to real data, where a second transfer-learning training step can take place. The spectrum has :math:`N_{\rm pix} \sim 300,000` pixels and :math:`N_{\rm lines} \sim 5000` spectral lines.  The number of lines is set by the `prominence=` kwarg: lower produces more lines and higher (up to about 0.3) produces fewer lines.  
"""
from cmath import log
import math
from mimetypes import init
import torch
from torch import nn
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
import torch.optim as optim
from tqdm import trange


class LinearEmulator(nn.Module):
    r"""
    Model for cloning a precomputed synthetic spectrum in linear flux.
    
    :math:`\mathsf{S} \mapsto \mathsf{S}_{\rm clone}`

    Parameters
    ----------
    wl_native :  torch.tensor
        The vector of input wavelengths at native resolution and sampling 
    flux_native : torch.tensor or None
        The vector of continuum-flattened input fluxes.  If None, line-finding is skipped, init_state_dict is required, and the 
        optimize method does not work.
    prominence : int or None
        The threshold prominence for peak finding, defaults to 0.03.  Ignored if init_state_dict is provided.
    init_state_dict : dict
        A dictionary of model parameters to initialize the model with
    """

    def __init__(
        self, wl_native, flux_native=None, prominence=None, init_state_dict=None
    ):
        super().__init__()

        # Read in the synthetic spectra at native resolution
        self.wl_native = torch.tensor(wl_native)
        self.wl_min = wl_native.min()
        self.wl_max = wl_native.max()
        self.n_pix = len(wl_native)

        ## Set up "active area", where the region-of-interest is:
        ## Restrict the lines to the active region plus 30 A buffer region
        ## These are hardcoded, and care should be taken if changing them
        line_buffer = 30  # Angstroms
        active_buffer = 60  # Angstroms

        active_lower, active_upper = (
            self.wl_min + active_buffer,
            self.wl_max - active_buffer,
        )
        active_mask = (wl_native > active_lower) & (wl_native < active_upper)
        self.active_mask = torch.tensor(active_mask)

        self.wl_active = self.wl_native[active_mask]

        if flux_native is not None:
            self.flux_native = torch.tensor(flux_native)
            self.flux_active = self.flux_native[active_mask]
        else:
            self.flux_native = None
            self.flux_active = None

        # Set up line threshold, where lines are computed outside the active area
        line_threshold_lower, line_threshold_upper = (
            self.wl_min + line_buffer,
            self.wl_max - line_buffer,
        )

        if init_state_dict is not None:
            if prominence is not None:
                print(
                    "You have entered both an initial state dict and a prominence kwarg.  Discarding prominence kwarg in favor of state dict."
                )
            lam_centers = init_state_dict["lam_centers"]
            log_amps = init_state_dict["amplitudes"]
            log_sigma_widths = init_state_dict["sigma_widths"]
            log_gamma_widths = init_state_dict["gamma_widths"]

        elif init_state_dict is None and self.flux_native is not None:
            if prominence is None:
                prominence = 0.03
            (lam_centers, amplitudes, widths_angstroms,) = self.detect_lines(
                self.wl_native, self.flux_native, prominence=prominence
            )

            # Experimentally determined scale factor tweaks
            amp_tweak = 0.14
            sigma_width_tweak = 1.28
            gamma_width_tweak = 1.52

            mask = (lam_centers > line_threshold_lower) & (
                lam_centers < line_threshold_upper
            )
            lam_centers = lam_centers[mask]
            log_amps = torch.log(amplitudes[mask] * amp_tweak)
            log_sigma_widths = torch.log(
                widths_angstroms[mask] / math.sqrt(2) * sigma_width_tweak
            )
            log_gamma_widths = torch.log(
                widths_angstroms[mask] / math.sqrt(2) * gamma_width_tweak
            )
        elif init_state_dict is None and self.flux_native is None:
            raise ValueError(
                "Either flux_native or init_state_dict must be provided to specify the spectral lines"
            )

        # Fix the wavelength centers as gospel for now.
        self.lam_centers = nn.Parameter(
            lam_centers.clone().detach().requires_grad_(False)
        )
        self.amplitudes = nn.Parameter(log_amps.clone().detach().requires_grad_(True))
        self.sigma_widths = nn.Parameter(
            log_sigma_widths.clone().detach().requires_grad_(True)
        )

        self.gamma_widths = nn.Parameter(
            log_gamma_widths.clone().detach().requires_grad_(True)
        )

        self.n_lines = len(lam_centers)

        self.a_coeff = nn.Parameter(
            torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        )
        self.b_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=False, dtype=torch.float64)
        )
        self.c_coeff = nn.Parameter(
            torch.tensor(0.0, requires_grad=False, dtype=torch.float64)
        )

        self.wl_normed = (self.wl_native - 10_500.0) / 2500.0

    def forward(self, wl):
        r"""The forward pass of the `blase` clone model

        Conducts the product of PseudoVoigt profiles for each line, with
        between one and three tunable parameters.  The entire spectrum can
        optionally be modulated by a tunable continuum polynomial.

        .. math:: 
            
            \mathsf{S}_{\rm clone} = \mathsf{P}(\lambda_S) \prod_{j=1}^{N_{\mathrm{lines}}} 1-a_j \mathsf{V}_j(\lambda_S)

        Parameters
        ----------
        wl : torch.tensor
            The input wavelength :math:`\mathbf{\lambda}_S` at which to 
            evaluate the model

        Returns
        -------
        torch.tensor
            The 1D generative spectral model clone :math:`\mathsf{S}_{\rm clone}` destined for backpropagation parameter tuning 
        """
        wl_normed = (wl - 10_500.0) / 2500.0

        polynomial_term = (
            self.a_coeff + self.b_coeff * wl_normed + self.c_coeff * wl_normed ** 2
        )

        return self.product_of_pseudovoigt_model(wl) * polynomial_term

    def product_of_pseudovoigt_model(self, wl):
        r"""Return the Product of pseudo-Voigt profiles
        
        The product acts like a matrix contraction:

        .. math:: \prod_{j=1}^{N_{\mathrm{lines}}} 1-a_j \mathsf{V}_j(\lambda_S)

        Parameters
        ----------
        wl : torch.tensor
            The input wavelength :math:`\mathbf{\lambda}_S` at which to 
            evaluate the model

        Returns
        -------
        torch.tensor
            The 1D generative spectral model clone :math:`\mathsf{S}_{\rm clone}` 
            destined for backpropagation parameter tuning     
        """
        return (1 - self.pseudo_voigt_profiles(wl)).prod(0)

    def detect_lines(self, wl_native, flux_native, prominence=0.03):
        """Identify the spectral lines in the native model

        Parameters
        ----------
        wl_native : torch.tensor
            The 1D vector of native model wavelengths (Angstroms)
        flux_native: torch.tensor
            The 1D vector of continuum-flattened model fluxes

        Returns
        -------
        tuple of tensors 
            The wavelength centers, prominences, and widths for all ID'ed 
            spectral lines
        """
        peaks, _ = find_peaks(-flux_native, distance=4, prominence=prominence)
        prominence_data = peak_prominences(-flux_native, peaks)
        width_data = peak_widths(-flux_native, peaks, prominence_data=prominence_data)
        lam_centers = wl_native[peaks]
        prominences = torch.tensor(prominence_data[0])
        widths = width_data[0]
        d_lam = np.diff(wl_native)[peaks]
        # Convert FWHM in pixels to Gaussian sigma in Angstroms
        widths_angs = torch.tensor(widths * d_lam / 2.355)

        return (lam_centers, prominences, widths_angs)

    def _lorentzian_line(self, lam_center, width, wavelengths):
        """Return a Lorentzian line, given properties"""
        return 1 / 3.141592654 * width / (width ** 2 + (wavelengths - lam_center) ** 2)

    def _gaussian_line(self, lam_center, width, wavelengths):
        """Return a normalized Gaussian line, given properties"""
        return (
            1.0
            / (width * 2.5066)
            * torch.exp(-0.5 * ((wavelengths - lam_center) / width) ** 2)
        )

    def _compute_eta(self, fwhm_L, fwhm):
        """Compute the eta mixture ratio for pseudo-Voigt weighting"""
        f_ratio = fwhm_L / fwhm
        return 1.36603 * f_ratio - 0.47719 * f_ratio ** 2 + 0.11116 * f_ratio ** 3

    def _compute_fwhm(self, fwhm_L, fwhm_G):
        """Compute the fwhm for pseudo Voigt using the approximation
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
        r"""Compute the pseudo-Voigt Profile for a collection of lines
        
        Much faster than the exact Voigt profile, but not as accurate:

        .. math:: 
        
            \mathsf{V}(\lambda_S-\lambda_{\mathrm{c},j}, \sigma_j, \gamma_j)

        
        Parameters
        ----------
        wavelengths : torch.tensor
            The 1D vector of wavelengths :math:`\mathbf{\lambda}_S` at which to 
            evaluate the model

        Returns
        -------
        torch.tensor
            The 1D pseudo-Voigt profiles

        Notes
        -----
        The pseudo-Voigt [1]_ is an approximation to the convolution of a 
        Lorentzian profile :math:`L(\lambda,f)` and Gaussian profile :math:`G(\lambda,f)`

        .. math::  V_p(\lambda,f) = \eta \cdot L(\lambda, f) + (1 - \eta) \cdot G(\lambda,f) 
        
        with mixture pre-factor:

        .. math::  \eta = 1.36603 (f_L/f) - 0.47719 (f_L/f)^2 + 0.11116(f_L/f)^3
        
        and FWHM:

        .. math::  f = [f_G^5 + 2.69269 f_G^4 f_L + 2.42843 f_G^3 f_L^2 + 4.47163 f_G^2 f_L^3 + 0.07842 f_G f_L^4 + f_L^5]^{1/5}


        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Pseudo-Voigt_profile
        """
        fwhm_G = 2.3548 * torch.exp(self.sigma_widths).unsqueeze(1)
        fwhm_L = 2.0 * torch.exp(self.gamma_widths).unsqueeze(1)
        fwhm = self._compute_fwhm(fwhm_L, fwhm_G)
        eta = self._compute_eta(fwhm_L, fwhm)

        return torch.exp(self.amplitudes).unsqueeze(1) * (
            eta
            * self._lorentzian_line(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.gamma_widths).unsqueeze(1),
                wavelengths.unsqueeze(0),
            )
            + (1 - eta)
            * self._gaussian_line(
                self.lam_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                wavelengths.unsqueeze(0),
            )
        )

    def optimize(self):
        """Optimize the model parameters"""
        raise NotImplementedError


class SparseLinearEmulator(LinearEmulator):
    r"""
    A sparse implementation of the LinearEmulator

    Parameters
    ----------
    wl_native : float vector
        The input wavelength at native sampling
    flux_native : float vector 
        The continuum-flattened flux at native sampling
    prominence : int
        The threshold for detecting lines
    device : Torch Device or str
        GPU or CPU?
    wing_cut_pixels : int
        The number of pixels centered on the line center to evaluate in the 
        sparse implementation, default: 1000 pixels
    init_state_dict : dict
        A dictionary of model parameters to initialize the model with
    """

    def __init__(
        self,
        wl_native,
        flux_native=None,
        prominence=None,
        device=None,
        wing_cut_pixels=None,
        init_state_dict=None,
    ):
        super().__init__(
            wl_native,
            flux_native=flux_native,
            prominence=prominence,
            init_state_dict=init_state_dict,
        )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        device = torch.device(device)

        if self.flux_native is not None:
            self.target = torch.tensor(
                self.flux_active, dtype=torch.float64, device=device
            )
        else:
            self.target = None

        ## Define the wing cut
        # Currently defined in *pixels*
        if wing_cut_pixels is None:
            wing_cut_pixels = 1000
        else:
            wing_cut_pixels = int(wing_cut_pixels)

        lines = self.lam_centers.detach().cpu().numpy()
        wl_native = self.wl_native.cpu().numpy()
        print("Initializing a sparse model with {:} spectral lines".format(len(lines)))

        # Find the index position of each spectral line
        center_indices = np.searchsorted(wl_native, lines)

        # From that, determine the beginning and ending indices
        zero_indices = center_indices - (wing_cut_pixels // 2)
        too_low = zero_indices < 0
        zero_indices[too_low] = 0
        end_indices = zero_indices + wing_cut_pixels
        too_high = end_indices > self.n_pix
        zero_indices[too_high] = self.n_pix - wing_cut_pixels - 1
        end_indices[too_high] = self.n_pix - 1

        # Make a 2D array of the indices
        indices_2D = np.linspace(
            zero_indices, end_indices, num=wing_cut_pixels, endpoint=True
        )

        self.indices_2D = torch.tensor(indices_2D.T, dtype=torch.long, device=device)
        self.indices_1D = self.indices_2D.reshape(-1)
        self.indices = self.indices_1D.unsqueeze(0)
        self.wl_2D = self.wl_native.to(device)[self.indices_2D]
        self.wl_1D = self.wl_2D.reshape(-1)
        self.active_mask = self.active_mask.to(device)

        self.radial_velocity = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
        )

    def forward(self):
        """The forward pass of the sparse implementation--- no wavelengths needed!

        Returns:
        torch.tensor
            The 1D generative spectral model destined for backpropagation
        """
        return self.sparse_pseudo_Voigt_model()

    def sparse_gaussian_model(self):
        """A sparse Gaussian-only model

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
            self.indices, ln_term, size=(self.n_pix,), requires_grad=True
        )

        result_1D = sparse_matrix.to_dense()

        return torch.exp(result_1D)

    def sparse_pseudo_Voigt_model(self):
        r"""A sparse pseudo-Voigt model

        The sparse matrix :math:`\hat{F}` is composed of the log flux 
        values.  Instead of a dense matrix  :math:`\bar{F}`, the log fluxes 
        are stored as trios of coordinate values and fluxes.  
        :math:`(i, j, \ln{F_{ji}})`.  The computation proceeds as follows:

        .. math::
        
            \mathsf{S}_{\rm clone} = \exp{\Big(\sum_{j=1}^{N_{lines}} \ln{F_{ji}} \Big)}

        Returns
        -------
        torch.tensor
            The 1D generative sparse spectral model 
        """
        fwhm_G = 2.3548 * torch.exp(self.sigma_widths).unsqueeze(1)
        fwhm_L = 2.0 * torch.exp(self.gamma_widths).unsqueeze(1)
        fwhm = self._compute_fwhm(fwhm_L, fwhm_G)
        eta = self._compute_eta(fwhm_L, fwhm)

        rv_shifted_centers = self.lam_centers * (
            1.0 + self.radial_velocity / 299_792.458
        )

        flux_2D = torch.exp(self.amplitudes).unsqueeze(1) * (
            eta
            * self._lorentzian_line(
                rv_shifted_centers.unsqueeze(1),
                torch.exp(self.gamma_widths).unsqueeze(1),
                self.wl_2D,
            )
            + (1 - eta)
            * self._gaussian_line(
                rv_shifted_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                self.wl_2D,
            )
        )

        # Enforce that you cannot have negative flux or emission lines
        flux_2D = torch.clamp(flux_2D, min=0.0, max=0.999999999)

        flux_1D = flux_2D.reshape(-1)
        ln_term = torch.log(1 - flux_1D)

        sparse_matrix = torch.sparse_coo_tensor(
            self.indices, ln_term, size=(self.n_pix,), requires_grad=True
        )

        result_1D = sparse_matrix.to_dense()

        return torch.exp(result_1D)

    def optimize(self, epochs=100, LR=0.01):
        """Optimize the model parameters with backpropagation
        
        Parameters
        ----------
        epochs : int
            The number of epochs to run the optimization for
        LR : float
            The learning rate for the optimizer

        Returns
        -------
        None
        """

        loss_fn = nn.MSELoss(reduction="mean")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), LR, amsgrad=True,
        )

        if self.target is None:
            raise (
                ValueError(
                    "No target spectrum provided, cannot optimize.  Initialize the model with the flux_native argument."
                )
            )

        t_iter = trange(epochs, desc="Training", leave=True)
        for epoch in t_iter:
            self.train()
            high_res_model = self.forward()[self.active_mask]
            loss = loss_fn(high_res_model, self.target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t_iter.set_description("Training Loss: {:0.8f}".format(loss.item()))


class ExtrinsicModel(nn.Module):
    r"""
    A Model for Extrinsic modulation


    Parameters
    ----------
    wl_native : float vector
        The native wavelength coordinates
    device : Torch Device or str
        GPU or CPU?
    """

    def __init__(self, wl_native, device=None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        device = torch.device(device)

        self.median_wl = np.median(wl_native)

        self.ln_vsini = nn.Parameter(
            torch.tensor(2.89, requires_grad=True, dtype=torch.float64)
        )

        # Make a fine wavelength grid from -4.5 to 4.5 Angstroms for convolution
        self.kernel_grid = torch.arange(
            -4.5, 4.51, 0.01, dtype=torch.float64, device=device
        )

    def forward(self, high_res_model):
        r"""The forward pass of the data-based echelle model implementation
        
        Computes the RV and vsini modulations of the native model:

        .. math::

            \mathsf{S}_{\rm ext}(\lambda_S) = \mathsf{S}_{\rm clone}(\lambda_\mathrm{c} - \frac{RV}{c}\lambda_\mathrm{c}) * \zeta \left(\frac{v}{v\sin{i}}\right)


        Parameters
        ----------
        high_res_model : torch.tensor
            The high resolution model fluxes sampled at the native wavelength grid

        Returns
        -------
        torch.tensor
            The high resolution model modulated for extrinsic parameters, :math:`\mathsf{S}_{\rm ext}`
        """
        vsini = 0.9 + torch.exp(self.ln_vsini)  # Floor of 0.9 km/s for now...
        rotationally_broadened = self.rotational_broaden(high_res_model, vsini)

        return rotationally_broadened

    def rotational_broaden(self, input_flux, vsini):
        r"""Rotationally broaden the spectrum

        Computes the convolution of the input flux with a Rotational
        Broadening kernel:

        .. math::

            \mathsf{S}_{\rm clone} * \zeta \left(\frac{v}{v\sin{i}}\right)
            

        Parameters
        ----------
        input_flux : torch.tensor
            The input flux vector, sampled at the native wavelength grid
        vsini : float scalar
            The rotational velocity in km/s

        Returns
        -------
        torch.tensor
            The rotationally broadened flux vector
        """
        velocity_grid = 299792.458 * self.kernel_grid / self.median_wl
        x = velocity_grid / vsini
        x2 = x * x
        x2 = torch.clamp(x2, max=1)
        kernel = torch.where(x2 < 0.99999999, 2.0 * torch.sqrt(1.0 - x2), 0.0)
        kernel = kernel / torch.sum(kernel)

        output = torch.nn.functional.conv1d(
            input_flux.unsqueeze(0).unsqueeze(1),
            kernel.unsqueeze(0).unsqueeze(1),
            padding="same",
        )
        return output.squeeze()


class InstrumentalModel(nn.Module):
    r"""
    A Model for instrumental resolution, etc (e.g. for a spectrograph)


    Parameters
    ----------
    wl_bin_edges : float vector
        The edges of the wavelength bins
    wl_native : float vector
        The native wavelength coordinates
    device : Torch Device or str
        GPU or CPU?
    """

    def __init__(self, wl_bin_edges, wl_native, device=None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        device = torch.device(device)

        self.wl_bin_edges = wl_bin_edges
        self.wl_centers = (wl_bin_edges[1:] + wl_bin_edges[:-1]) / 2.0
        self.median_wl = np.median(self.wl_centers)
        self.bandwidth = self.wl_bin_edges[-1] + self.wl_bin_edges[0]

        # self.resolving_power = nn.Parameter(
        #    torch.tensor(45_000.0, requires_grad=True, dtype=torch.float64)
        # )
        self.ln_sigma_angs = nn.Parameter(
            torch.tensor(-2.8134, requires_grad=True, dtype=torch.float64)
        )

        # Make a fine wavelength grid from -4.5 to 4.5 Angstroms for convolution
        self.kernel_grid = torch.arange(
            -4.5, 4.51, 0.01, dtype=torch.float64, device=device
        )

        labels = np.searchsorted(wl_bin_edges, wl_native)
        indices = torch.tensor(labels)
        _idx, vals = torch.unique(indices, return_counts=True)
        self.label_spacings = tuple(vals)

        # Polynomial coefficients for the continuum
        # For this example, the output y is a linear function of (x, x^2, x^3... x^p), so
        # we can consider it as a linear layer neural network. Let's prepare the
        # tensor (x, x^2, x^3, ... x^p).
        max_p = 15
        p_exponents = torch.arange(1, max_p + 1, device=device)

        self.wl_normed = torch.tensor(
            (self.wl_centers - self.median_wl) / self.bandwidth,
            device=device,
            dtype=torch.float64,
        )
        self.design_matrix = (
            self.wl_normed.unsqueeze(-1).pow(p_exponents).to(torch.float64)
        )
        self.linear_model = torch.nn.Linear(
            max_p, 1, device=device, dtype=torch.float64
        )
        self.linear_model.weight = torch.nn.Parameter(
            torch.zeros((1, max_p), dtype=torch.float64)
        )
        self.linear_model.bias = torch.nn.Parameter(
            torch.tensor([1.0], dtype=torch.float64)
        )

    def forward(self, high_res_model):
        r"""The forward pass of the instrumental model
        
        Computes the instrumental modulation of the joint model and resamples the spectrum to the coarse data wavelength coordinates.

        We start with a joint model composed of the elementwise product of the extrinsic stellar spectrum with the resampled telluric spectrum:

        .. math::  \mathsf{M}_{\rm joint} = \mathsf{S}_{\rm ext} \odot \mathsf{T}(\lambda_S) \\ 

        An intermediate high resolution spectrum is computed by convolving the joint model with a Gaussian kernel, and weighting by a smooth polynomial shape:

        .. math::  \mathsf{M}_{\rm inst}(\lambda_S) = \mathsf{P} \odot \Big(\mathsf{M}_{\rm joint} * g(R) \Big)

        Finally, we resample the intermediate high resolution instrumental spectrum to the coarser data wavelength coordinates:

        .. math::  \mathsf{M}(\lambda_D) = \text{resample} \Big[ \mathsf{M}_{\rm inst}(\lambda_S) \Big]


        Parameters
        ----------
        high_res_model : torch.tensor
            The high resolution model fluxes sampled at the native wavelength grid.  The high resolution model is typically the joint model including the extrinsic spectrum and telluric spectrum, :math:`\mathsf{M}_{\rm joint}`.  It can alternatively be a bare extrinisic spectrum, :math:`\mathsf{S}_{\rm ext}` if telluric absorption is negligible in the wavelength range of interest.

        Returns
        -------
        torch.tensor
            The high resolution model modulated for extrinsic parameters, and resampled at the data coordinates :math:`\mathsf{M}(\lambda_D)`.
        """
        sigma_angs = 0.01 + torch.exp(self.ln_sigma_angs)  # Floor of 0.01 Angstroms...
        convolved_flux = self.instrumental_broaden(high_res_model, sigma_angs)
        resampled_flux = self.resample_to_data(convolved_flux)

        return resampled_flux * self.warped_continuum()

    def warped_continuum(self):
        """Warp the continuum by a smooth polynomial"""
        return self.linear_model(self.design_matrix).squeeze()

    def resample_to_data(self, convolved_flux):
        """Resample the high resolution model to the data wavelength sampling"""
        vs = torch.split_with_sizes(convolved_flux, self.label_spacings)
        resampled_model_flux = torch.stack([torch.mean(v) for v in vs])
        resampled_model_flux = torch.clamp(resampled_model_flux, min=0.0, max=1.0)

        # Discard the first and last bins outside the spectrum extents
        return resampled_model_flux[1:-1]

    def instrumental_broaden(self, input_flux, sigma_angs):
        """Instrumental broaden the spectrum

        Parameters
        ----------
        input_flux : torch.tensor
            The input flux vector to be broadened
        sigma_angs : float scalar 
            The spectral resolution sigma in Angstroms

        Returns
        -------
        torch.tensor
            The instrumental broadened flux vector
        """
        weights = (
            1
            / (sigma_angs * torch.sqrt(torch.tensor(2 * 3.1415926654)))
            * torch.exp(-1.0 / 2.0 * self.kernel_grid ** 2 / sigma_angs ** 2)
        ) * 0.01  # kernel step size!

        output = torch.nn.functional.conv1d(
            input_flux.unsqueeze(0).unsqueeze(1),
            weights.unsqueeze(0).unsqueeze(1),
            padding="same",
        )
        return output.squeeze()


class SparseLogEmulator(SparseLinearEmulator):
    r"""
    A log version of the sparse emulator

    Parameters
    ----------
    wl_native : float vector
        The input wavelength at native sampling
    lnflux_native : float vector 
        The natural log of the continuum-flattened flux at native sampling
    prominence : int
        The threshold for detecting lines
    device : Torch Device or str
        GPU or CPU?
    wing_cut_pixels : int
        The number of pixels centered on the line center to evaluate in the 
        sparse implementation, default: 1000 pixels
    init_state_dict : dict
        The initial state of the model
    """

    def __init__(
        self,
        wl_native,
        lnflux_native,
        prominence=0.01,
        device=None,
        wing_cut_pixels=None,
        init_state_dict=None,
    ):
        super().__init__(
            wl_native,
            lnflux_native,
            prominence=prominence,
            device=device,
            wing_cut_pixels=wing_cut_pixels,
            init_state_dict=init_state_dict,
        )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        device = torch.device(device)

        # The clone-native comparison is done in linear space:
        self.target = torch.exp(torch.tensor(lnflux_native, device=device))[
            self.active_mask.to(device)
        ].to(device)

    def forward(self):
        """The forward pass of the sparse implementation--- no wavelengths needed!

        Returns:
        torch.tensor
            The 1D generative spectral model destined for backpropagation
        """
        return self.sparse_opacity_model()

    def sparse_opacity_model(self):
        r"""A sparse pseudo-Voigt model

        The sparse matrix :math:`\hat{F}` is composed of the log flux 
        values.  Instead of a dense matrix  :math:`\bar{F}`, the log fluxes 
        are stored as trios of coordinate values and fluxes.  
        :math:`(i, j, \ln{F_{ji}})`.  The computation proceeds as follows:

        .. math::
        
            \mathsf{S}_{\rm clone} = \exp{\Big(-\sum_{j=1}^{N_{\mathrm{lines}}} a_j \mathsf{V}_j\Big)}

        Returns
        -------
        torch.tensor
            The 1D generative sparse spectral model 
        """
        fwhm_G = 2.3548 * torch.exp(self.sigma_widths).unsqueeze(1)
        fwhm_L = 2.0 * torch.exp(self.gamma_widths).unsqueeze(1)
        fwhm = self._compute_fwhm(fwhm_L, fwhm_G)
        eta = self._compute_eta(fwhm_L, fwhm)

        rv_shifted_centers = self.lam_centers * (
            1.0 + self.radial_velocity / 299_792.458
        )

        opacities_2D = torch.exp(self.amplitudes).unsqueeze(1) * (
            eta
            * self._lorentzian_line(
                rv_shifted_centers.unsqueeze(1),
                torch.exp(self.gamma_widths).unsqueeze(1),
                self.wl_2D,
            )
            + (1 - eta)
            * self._gaussian_line(
                rv_shifted_centers.unsqueeze(1),
                torch.exp(self.sigma_widths).unsqueeze(1),
                self.wl_2D,
            )
        )

        opacities_1D = opacities_2D.reshape(-1)
        negative_opacities = -1 * opacities_1D

        sparse_matrix = torch.sparse_coo_tensor(
            self.indices, negative_opacities, size=(self.n_pix,), requires_grad=True
        )

        result_1D = sparse_matrix.to_dense()

        return torch.exp(result_1D)


# Deprecated class names
PhoenixEmulator = LinearEmulator
SparsePhoenixEmulator = SparseLinearEmulator
EchelleModel = ExtrinsicModel
