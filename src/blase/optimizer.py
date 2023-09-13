import gc
import torch

from blase.emulator import SparseLinearEmulator as SLE
from gollum.phoenix import *
from typing import Callable

extract = lambda x: x.detach().cpu().numpy()


def default_clean(spec: PHOENIXSpectrum):
    norm_spec = spec.divide_by_blackbody().normalize()
    continuum = norm_spec.fit_continuum(polyorder=5)
    return norm_spec.divide(continuum, handle_meta="ff")


def run_emulator(
    spec: PHOENIXSpectrum,
    preprocessor: Callable = default_clean,
    wing_cut: int = 6000,
    prominence: float = 0.005,
    epochs: int = 100,
    LR: float = 0.05,
    device: str = "cpu",
):
    file_name = f"T{spec.teff}G{spec.logg}Z{spec.Z}.pt"
    if os.path.exists(f"emulator_states/{file_name}"):
        return
    if not os.path.exists("emulator_states"):
        os.mkdir("emulator_states")
    # Preprocessing step
    clean_spec = preprocessor(spec)
    # Create emulator
    emulator = SLE(
        clean_spec.wavelength.value, clean_spec.flux.value, prominence, device, wing_cut
    )
    pre_line_centers = emulator.lam_centers.clone().detach()
    emulator.to(device)
    emulator.optimize(epochs=epochs, LR=LR)
    # Write state dict to .pt file
    state_dict = emulator.state_dict()
    state_dict['pre_line_centers'] = pre_line_centers
    torch.save(state_dict, f"emulator_states/{file_name}")
    # Tie up loose ends
    del emulator
    gc.collect()
    torch.cuda.empty_cache()
