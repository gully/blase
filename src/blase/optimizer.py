import gc
import torch

from blase.emulator import SparseLinearEmulator as SLE
from gollum.phoenix import *
from sys import argv

extract = lambda x: x.detach().cpu().numpy()

def run_emulator(spec: PHOENIXSpectrum, 
                 wing_cut: int = 6000, 
                 prominence: float = 0.005, 
                 epochs: int = 100, 
                 LR: float = 0.05, 
                 device: str = 'cpu'):
    # Preprocessing step
    norm_spec = spec.divide_by_blackbody().normalize()
    continuum = norm_spec.fit_continuum(polyorder=5)
    clean_spec = norm_spec.divide(continuum, handle_meta='ff')
    # Create emulator
    emulator = SLE(clean_spec.wavelength.value, clean_spec.flux.value, prominence, device, wing_cut)
    emulator.to(device)
    emulator.optimize(epochs=epochs, LR=LR)
    # Write state dict to .pt file
    torch.save(emulator.state_dict(), f'emulator_states/T{spec.teff}G{spec.logg}Z{spec.Z}.pt')
    # Tie up loose ends
    del emulator
    gc.collect()
    torch.cuda.empty_cache()