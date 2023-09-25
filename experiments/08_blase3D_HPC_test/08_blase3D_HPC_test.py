import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.optimizer import default_clean, run_emulator
from gollum.phoenix import PHOENIXGrid
from sys import argv
from tqdm import tqdm


def main(device):
    grid = PHOENIXGrid(wl_lo=10000, wl_hi=11000, path="/data/libraries/raw/PHOENIX/")
    for spec in grid:
        run_emulator(spec, device=device)

if __name__ == "__main__":
    try:
        device = argv[1]
    except IndexError:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                device = torch.device("mps")
            except RuntimeError:
                device = torch.device("cpu") 
    main(device)