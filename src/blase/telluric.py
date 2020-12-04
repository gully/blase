"""
telluric
--------

Telluric absorption forward modeling based on HITRAN

TelluricModel
#############
"""

import torch
from torch import nn
from blase.utils import suppress_stdout
import math
from collections import OrderedDict

with suppress_stdout():
    import hapi

    hapi.db_begin("../../hapi/data/")

# custom dataset loader
class TelluricModel(nn.Module):
    r"""Make a model of Earth's atmospheric absorption and/or sky emission

    Args:
        device (str): On which device to run the model, "cuda" or "cpu"

    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.hitran_columns = [
            "n_air",
            "gamma_air",
            "gamma_self",
            "elower",
            "gpp",
            "nu",
            "delta_air",
            "sw",
        ]
        self.hitran_c2 = 1.4387770  # cm K

    def forward(self):
        """The forward pass of the neural network"""
        return torch.ones(2048, device=self.device, dtype=torch.float64)

    def get_hapi_molec_data(self, species):
        r"""Fetch HITRAN atomic and molecular data as torch tensors

        Args:
            species (str): Which atomic/molecular species to examine

        Returns:
            dict: A dictionary containing tensors of size :math:`N_{\mathrm{lines}}` 
                for each of the 8 HITRAN columns of interest
        """

        out_dict = OrderedDict(
            (
                col,
                torch.tensor(
                    hapi.getColumn(species, col),
                    dtype=torch.float64,
                    device=self.device,
                ),
            )
            for col in self.hitran_columns
        )
        return out_dict

    def gamma_of_p_and_T(self, p, T, p_self, n_air, gamma_air_ref, gamma_self_ref):
        r"""Compute the Lorentz half width at half maximum (HWHM) in units of :math:`\mathrm{cm^{-1}}`
        with pressure and temperature: 
        
        .. math::

            \gamma(p, T) = \left( \frac{T_\mathrm{ref}}{T} \right)^{n_\mathrm{air}}\left( \gamma_\mathrm{air}(p_\mathrm{ref}, T_\mathrm{ref})(p-p_\mathrm{self}) + \gamma_\mathrm{self}(p_\mathrm{ref}, T_\mathrm{ref})p_\mathrm{self}\right)


        Args:
            p (float): Pressure :math:`p` in standard atmospheres `atm`
            T (float): Temperature :math:`T` in `K`
            p_self (float): Partial pressure of the species in `atm`
            n_air (float): The coefficient of the temperature dependence of the air-broadened 
                half width (dimensionless)
            gamma_air_ref (float): The air-broadened HWHM at :math:`T_{ref}=296\;` K  and 
                reference pressure :math:`p_{ref}=1\;` atm
            gamma_self_ref (float): The self-broadened HWHM at :math:`T_{ref}=296\;` K  
                and reference pressure :math:`p_{ref}=1\;` atm

        Returns:
            torch.Tensor: A vector of length :math:`N_{\mathrm{lines}}` 

        """

        return (296.0 / T) ** n_air * (
            gamma_air_ref * (p - p_self) + gamma_self_ref * (p_self)
        )

    def lorentz_profile(self, nu, p, nu_ij, gamma, dp_ref, S_ij):
        r"""Return the Lorentz line profile given vectors and parameters

        .. math::
        
            f_\mathrm{L}(\nu; \nu_{ij}, T, p) = \frac{1}{\pi}\frac{\gamma(p,T)}{\gamma(p,T)^2 + [\nu-(\nu_{ij} + \delta(p_\mathrm{ref})p)]^2}    

        Args:
            nu (float): Wavenumber variable input :math:`\nu` in :math:`\mathrm{cm^{-1}}`.
                For matrix output, nu should have shape :math:`N_{\lambda} \times 1`.
            p (float): Pressure :math:`p` in standard atmospheres `atm`
            nu_ij (float): Wavenumber of the spectral line transition :math:`(\mathrm{cm^{-1}})` in vacuum
            gamma (float): Lorentz half width at half maximum (HWHM), :math:`\gamma` in units of :math:`\mathrm{cm^{-1}}`
            dp_ref (float): The pressure shift :math:`\mathrm{cm^{-1}/atm}` at :math:`T_{ref}=296` K
                and :math:`p_{ref} = 1` atm of the line position with respect to the vacuum transition 
                wavenumber :math:`\nu_{ij}`
            S_ij (float): The spectral line intensity :math:`\mathrm{cm^{-1}/(molecule·cm^{-2}})`
            
        Returns:
            torch.Tensor: Either a vector of length :math:`N_\lambda` if :math:`\gamma` is a scalar, or a matrix of size :math:`N_\lambda \times N_{lines}` if :math:`\gamma` is a vector
        
        """
        return S_ij / math.pi * gamma / (gamma ** 2 + (nu - (nu_ij + dp_ref * p)) ** 2)

    def tips_Q_of_T(self, T, g_k, E_k):
        r"""Total Internal Partition Sum
        
        .. math :: 
        
            Q(T) = \sum_k g_k \exp\left(-\frac{c_2E_k}{T}\right)


        Args:
            T (float): Temperature :math:`T` in `K`
            g_k (float): The lower state statistical weights :math:`g_k`
            E_k (float): The lower-state energy of the transition :math:`\mathrm{cm^{-1}}`
        
        Returns:
            torch.Tensor: A scalar or a vector the same length as T    
        
        """
        return torch.sum(g_k * torch.exp(-self.hitran_c2 * E_k / T))

    def S_ij_of_T(self, T, S_ij_296, nu_ij, g_lower, E_lower):
        r"""The Spectral Line Intensity as a function of temperature
        
        .. math::

            S_{ij}(T) = S_{ij}(T_\mathrm{ref}) \frac{Q(T_\mathrm{ref})}{Q(T)} \frac{\exp\left( -c_2 E''/T \right)}{\exp\left( -c_2 E''/T_\mathrm{ref} \right)} \frac{[1-\exp\left( -c_2 \nu_{ij}/T \right)]}{[1-\exp\left(-c_2 \nu_{ij}/T_\mathrm{ref} \right)]}

        Args:
            T (float): Temperature :math:`T` in `K`
            S_ij_296 (float): The spectral line intensity at :math:`T_{ref}=296` K
            nu_ij (float): Wavenumber of the spectral line transition :math:`(\mathrm{cm^{-1}})` in vacuum
            g_lower (float): The lower state statistical weights :math:`g''`
            E_lower (float): The lower-state energy of the transition :math:`\mathrm{cm^{-1}}`
        
        Returns:
            torch.Tensor: A vector of length :math:`N_{\mathrm{lines}}`  

        """
        c_2 = 1.4387770  # cm K
        with torch.no_grad():
            Q_at_296 = self.tips_Q_of_T(296.0, g_lower, E_lower)
        return (
            S_ij_296
            * Q_at_296
            / self.tips_Q_of_T(T, g_lower, E_lower)
            * torch.exp(-c_2 * E_lower / T)
            / torch.exp(-c_2 * E_lower / 296.0)
            * (1 - torch.exp(-c_2 * nu_ij / T))
            / (1 - torch.exp(-c_2 * nu_ij / 296.0))
        )

    def transmission_of_T_p(self, T, p, nus, vol_mix_ratio, hitran):
        r"""Return the transmission spectrum :math:`\mathcal{T}(\nu; T, P)=\exp(-\tau_\nu)` for 1 km of pathlength

        Args:
            T (float): Temperature :math:`T` in `K`
            p (float): Pressure :math:`p` in standard atmospheres `atm`
            nus (float): Wavenumber variable input :math:`\nu` in :math:`\mathrm{cm^{-1}}`.
            vol_mix_ratio (float): The volume mixing ratio of the species assuming ideal gas
        
        Returns:
            torch.Tensor: A vector of length :math:`N_{\nu}`  
        
        """
        gammas = self.gamma_of_p_and_T(
            p,
            T,
            vol_mix_ratio,
            hitran["n_air"],
            hitran["gamma_air"],
            hitran["gamma_self"],
        )

        S_ij = self.S_ij_of_T(
            T, hitran["sw"], hitran["nu"], hitran["gpp"], hitran["elower"]
        )

        abs_coeff = self.lorentz_profile(
            nus.unsqueeze(1), p, hitran["nu"], gammas, hitran["delta_air"], S_ij
        ).sum(1)

        # path_length_km = 1.0
        tau = abs_coeff * (vol_mix_ratio * 2.688392857142857e19) * (1.0 * 100000.0)
        return torch.exp(-tau)

    def transmission_multilayer_atmosphere(
        self, T_vector, p_vector, nus, vol_mix_ratio, hitran
    ):
        r"""Return the transmission spectrum :math:`\mathcal{T}(\nu; T, P)=\exp(-\tau_\nu)` for 
        a cascade of :math:`N_{layers}` pathlengths with thickness 1 km.

        Args:
            T (:math:`1 \times 1 \times N_{layers}` tensor): Temperature :math:`T` in `K`
            p (:math:`1 \times 1 \times N_{layers}` tensor): Pressure :math:`p` in standard atmospheres `atm`
            nus (:math:`N_{\nu} \times 1 \times 1` tensor): Wavenumber variable input :math:`\nu` in :math:`\mathrm{cm^{-1}}`.
            vol_mix_ratio (scalar or :math:`1 \times 1 \times N_{layers}` tensor): The volume mixing ratio of the species assuming ideal gas
            hitran (OrderedDict): Each entry of consists of a :math:`N_{lines}` vector that will be broadcasted to
                a :math:`1 \times N_{lines} \times 1` tensor when operating with the other vectors.
        Returns:
            torch.Tensor: A vector of length :math:`N_{\nu}`  
        
        """
        return None
