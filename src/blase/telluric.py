"""
telluric
--------

Telluric absorption forward modeling based on HITRAN

TelluricModel
############
"""

import torch
from torch import nn
from blase.utils import suppress_stdout

with suppress_stdout():
    import hapi


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
