from datetime import datetime as dt
import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.optimizer import default_clean, run_emulator
from gollum.phoenix import PHOENIXGrid
from os import system
from sys import argv
from tqdm import tqdm


def main(wl_lo: int, wl_hi: int, device: str | torch.device):
    grid = PHOENIXGrid(wl_lo=10000, wl_hi=11000, path="/data/libraries/raw/PHOENIX/")
    for spec in grid:
        run_emulator(spec, device=device)

if __name__ == "__main__":
    start_time = dt.now()
    try:
        device = argv[3]
    except IndexError:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                device = torch.device("mps")
            except RuntimeError:
                device = torch.device("cpu") 
    main(argv[1], argv[2], device)
    end_time = dt.now()
    system(f"echo {f'{start_time} to {end_time}: {end_time - start_time}'} >> delta.txt")
