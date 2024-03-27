r"""
utilities
--------------

These general purpose methods are used in one or more other modules.
"""
import numpy as np
import torch

from astropy import constants as const
from astropy import units as u
from contextlib import contextmanager
from os import devnull
from sys import stdout


@contextmanager
def suppress_stdout():
    with open(devnull, "w") as devnull:
        old_stdout, stdout = stdout, devnull
        try:
            yield
        finally:
            stdout = old_stdout

def doppler_grid(lambda_0, lambda_max, velocity_resolution_kmps=0.5):
    """Compute the exponential sampling interval for a given lambda_0 and velocity resolution."""
    c_kmps = const.c.to(u.km / u.s).value
    velocity_max = c_kmps * np.log(lambda_max / lambda_0)
    velocity_vector = np.arange(0, velocity_max, velocity_resolution_kmps)
    return lambda_0 * np.exp(velocity_vector / c_kmps)

def auto_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        try:
            return torch.device("mps")
        except RuntimeError:
            return torch.device("cpu")
