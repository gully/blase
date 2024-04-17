import numpy as np
import pandas as pd
import sys
import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.optimizer import default_clean
from collections import defaultdict
from gollum.phoenix import PHOENIXSpectrum, PHOENIXGrid
from os import listdir
from pickle import dump, load
from re import split
from skopt import gp_minimize
from scipy.interpolate import RegularGridInterpolator
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

def create_interpolators(df: pd.DataFrame, df_gp: pd.DataFrame) -> list[RegularGridInterpolator]:
    interpolator_list = []
    for line in tqdm(df.center.unique()):
        df_line = df.query('center == @line', engine='python').merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
        df_line.sort_values(['teff', 'logg', 'Z'], inplace=True)
        interpolator_list.append(RegularGridInterpolator(
            (df_line.teff.unique(), df_line.logg.unique(), df_line.Z.unique()), 
            df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy().reshape(len(df_line.teff.unique()), len(df_line.logg.unique()), len(df_line.Z.unique()), 4),
            method='linear'))
    return interpolator_list

def reconstruct(wl_grid: np.ndarray, point: tuple[int, float, float], interpolator_list: list[RegularGridInterpolator]) -> np.ndarray:
    output = np.vstack([r for interpolator in interpolator_list if (r := interpolator([point[0], point[1], point[2]]).squeeze())[0] != -1000])
    state_dict = {
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    return np.nan_to_num(SLE(wl_native=wl_grid, init_state_dict=state_dict).forward().detach().numpy(), nan=1)

def loss_fn(wl_grid: np.ndarray, data: np.ndarray, interpolator_list: list[RegularGridInterpolator]) -> Callable:
    return lambda point: ((reconstruct(wl_grid, point, interpolator_list) - data)**2).mean()**0.5

def pickling_run():
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

def inference_test():
    from time import perf_counter
    interpolator_list = load(open('interpolator_list.pkl', 'rb'))
    start = perf_counter()
    spec = default_clean(PHOENIXSpectrum(teff=5000, logg=4, Z=0, download=True))
    res = gp_minimize(loss_fn(spec.wavelength.value, spec.flux.value, interpolator_list), dimensions=[(2300, 12000), (2, 6), (-0.5, 0)], n_calls=50, n_random_starts=30)
    print(f'Result: {res.x} achieved in {perf_counter() - start} s')

def inference(wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
    return np.array(gp_minimize(loss_fn(wavelength, flux), dimensions=[(2300, 12000), (2, 6), (-0.5, 0)], n_calls=50, n_random_starts=30).x)

def inference_grid():
    sys.stderr = open('log.txt', 'w')
    interpolator_list = load(open('interpolator_list.pkl', 'rb'))
    grid = PHOENIXGrid(teff_range=(2300, 12000), logg_range=(2, 6), Z_range=(-0.5, 0), path='/data/libraries/raw/PHOENIX/')
    df_list = []
    for point in grid.grid_points:
        spec = default_clean(grid[grid.lookup_dict[point]])
        result = np.vstack(inference(spec.wavelength.value, spec.flux.value) for _ in range(10))
        df_list.append(pd.DataFrame({'teff': point[0], 'logg': point[1], 'Z': point[2], 'i_teff': result[:, 0], 'i_logg': result[:, 1], 'i_Z': result[:, 2]}))
    pd.concat(df_list).to_parquet('inference_results.parquet.gz', compression='gzip')


if __name__ == '__main__':
    inference_test()
