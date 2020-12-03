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

    def forward(self):
        """The forward pass of the neural network"""
        return torch.ones(2048, device=self.device, dtype=torch.float64)

    def gamma_of_p_and_T(self, p, T, p_self, n_air, gamma_air_ref, gamma_self_ref):
        r"""Compute the Lorentz half width at half maximum (HWHM) in units of :math:`\mathrm{cm^{-1}/atm}`
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
        """

        return (296.0 / T) ** n_air * (
            gamma_air_ref * (p - p_self) + gamma_self_ref * (p_self)
        )
