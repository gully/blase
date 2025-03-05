from datetime import datetime as dt
import torch

from blase.optimizer import run_emulator
from gollum.phoenix import PHOENIXGrid
from os import system
from sys import argv


def main(wl_lo: int, wl_hi: int, device: str | torch.device):
    grid = PHOENIXGrid(wl_lo=wl_lo, wl_hi=wl_hi, path="/mnt/c/Users/sujay/Downloads/PHOENIX/")
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
    main(8038, 12849, device)
    end_time = dt.now()
    system(f"echo {f'{start_time} to {end_time}: {end_time - start_time}'} >> delta.txt")
