"""
datasets
--------

This utility loads 1D echelle data, inheriting from the PyTorch Dataset model.
We currently only support data from the HPF spectrograph.

HPFDataset
############
"""

import torch
from torch.utils.data import Dataset
from astropy.io import fits
import pandas as pd
import numpy as np


# custom dataset loader
class HPFDataset(Dataset):
    r"""Read in an HPF spectrum

    Args:
        filename (list): the Goldilocks echelle spectrum to read-in
        device (str): On which device to house the data, "cuda" or "cpu"

    """

    def __init__(self, filename, device="cuda"):
        super().__init__()
        self.filename = filename
        self.n_pix = 2048
        self.n_orders = 28
        self.n_slices = 9
        hdus = fits.open(filename)
        self.hdus = hdus

        data_cube = []
        for i in range(1, self.n_slices + 1):
            data_cube.append(hdus[i].data.astype(np.float64))

        self.data_cube = torch.tensor(data_cube)

        # normalize the flux dimension:
        nans = torch.isnan(self.data_cube)
        self.data_cube[nans] = 0.0  # for now...
        med, _ = torch.median(self.data_cube[0, :, :], axis=1)
        self.data_cube[0, :, :] = self.data_cube[0, :, :] / med.unsqueeze(1)

        self.df = self.get_goldilocks_dataframe(filename)

    def __getitem__(self, index):
        """The index represents the echelle order"""
        return (index, self.data_cube[0, index, :])

    def __len__(self):
        # Only do 3 orders right now
        return self.n_orders

    def get_goldilocks_dataframe(self, filename):
        """Return a pandas Dataframe given a Goldilocks FITS file name"""
        df = pd.DataFrame()
        for j in range(self.n_orders):
            df_i = pd.DataFrame()
            for i in range(1, 10):
                name = self.hdus[i].name
                df_i[name] = self.hdus[i].data[j, :]
            df_i["order"] = j
            df = df.append(df_i, ignore_index=True)
        keep_mask = df[df.columns[0:6]] != 0.0
        df = df[keep_mask.all(axis=1)].reset_index(drop=True)

        return df
