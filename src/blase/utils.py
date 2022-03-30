from contextlib import contextmanager
import sys, os
import numpy as np
from astropy import constants as const
from astropy import units as u


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def doppler_grid(lambda_0, lambda_max, velocity_resolution_kmps=0.5):
    """Compute the exponential sampling interval for a given lambda_0 and velocity resolution.
    """
    c_kmps = const.c.to(u.km / u.s).value
    velocity_max = c_kmps * np.log(lambda_max / lambda_0)
    velocity_vector = np.arange(0, velocity_max, velocity_resolution_kmps)
    return lambda_0 * np.exp(velocity_vector / c_kmps)

