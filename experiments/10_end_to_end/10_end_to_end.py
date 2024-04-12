import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.utils import doppler_grid
from collections import defaultdict
from functools import partial
from gollum.phoenix import PHOENIXSpectrum
from os import listdir
from pickle import dump, load
from re import split
from skopt import gp_minimize
from scipy.interpolate import griddata, RegularGridInterpolator
from tqdm import tqdm
from typing import Callable


def read_state_dicts(path: str) -> pd.DataFrame:
    line_stats = defaultdict(list)
    for f in tqdm(listdir(path)):
        state_dict = torch.load(f'{path}/{f}', map_location='cuda:0')
        tokens = split('[TGZ]', f[:-3])
        line_stats['teff'].append(int(tokens[1]))
        line_stats['logg'].append(float(tokens[2]))
        line_stats['Z'].append(float(tokens[3]))
        line_stats['center'].append(state_dict['pre_line_centers'].cpu().numpy())
        line_stats['shift_center'].append(state_dict['lam_centers'].cpu().numpy())
        line_stats['amp'].append(state_dict['amplitudes'].cpu().numpy())
        line_stats['sigma'].append(state_dict['sigma_widths'].cpu().numpy())
        line_stats['gamma'].append(state_dict['gamma_widths'].cpu().numpy())
    return pd.DataFrame(line_stats)

def optimize_memory(df: pd.DataFrame):
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

def create_interpolators(df: pd.DataFrame, df_gp: pd.DataFrame) -> list[partial]:
    interpolator_list = []
    for line in tqdm(df.center.unique()):
        df_line = df.query('center == @line', engine='python').merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
        df_line.sort_values(['teff', 'logg', 'Z'], inplace=True)
        interpolator_list.append(RegularGridInterpolator(
            (df_line.teff.unique(), df_line.logg.unique(), df_line.Z.unique()), 
            df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy().reshape(len(df_line.teff.unique()), len(df_line.logg.unique()), len(df_line.Z.unique()), 4),
            method='linear'))
    return interpolator_list

def reconstruct(wl_grid: np.ndarray, point: tuple[int, float, float]) -> np.ndarray:
    interpolators = load(open('interpolator_list.pkl', 'rb'))
    output = np.vstack([r for interpolator in interpolators if (r := interpolator([point[0], point[1], point[2]]).squeeze())[0] != -1000])
    state_dict = {
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    return np.nan_to_num(SLE(wl_native=wl_grid, init_state_dict=state_dict, device="cpu").forward().detach().numpy(), nan=1)

def loss_fn(wl_grid: np.ndarray, data: np.ndarray) -> Callable:
    return lambda point: ((reconstruct(wl_grid, point) - data)**2).mean()**0.5

def triton_run():
    sys.stderr = open('log.txt', 'w')
    path = '/home/sujays/github/blase/experiments/08_blase3D_HPC_test/emulator_states'
    df = read_state_dicts(path)
    df_gp = df[['teff', 'logg', 'Z']]
    df = df.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    interpolators = create_interpolators(df, df_gp)
    dump(interpolators, open('interpolator_list.pkl', 'wb'))
    with open('log.txt', 'w') as f:
        f.write('Interpolator partials dumped to pickle.')

def inference_run():
    from blase.optimizer import default_clean
    from time import perf_counter
    start = perf_counter()
    spec = default_clean(PHOENIXSpectrum(teff=5000, logg=4, Z=0, download=True))
    res = gp_minimize(loss_fn(spec.wavelength.value, spec.flux.value), dimensions=[(2300, 12000), (2, 6), (-0.5, 0)], n_calls=50, n_random_starts=30)
    print(f'Result: {res.x} achieved in {perf_counter() - start} s')

if __name__ == '__main__':
    inference_run()
