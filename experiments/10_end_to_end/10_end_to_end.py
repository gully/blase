import numpy as np
import pandas as pd
import sys
import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.optimizer import default_clean
from collections import defaultdict
from gollum.phoenix import PHOENIXSpectrum
from itertools import product
from os import listdir
from pickle import dump, load
from re import split
from skopt import gp_minimize
from scipy.interpolate import RegularGridInterpolator
from time import perf_counter
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
    for line in tqdm(df.value_counts('center').index):
        df_line = df.query('center == @line', engine='python').merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
        df_line.sort_values(['teff', 'logg', 'Z'], inplace=True)
        interpolator_list.append(RegularGridInterpolator(
            (df_line.teff.unique(), df_line.logg.unique(), df_line.Z.unique()), 
            df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy().reshape(len(df_line.teff.unique()), len(df_line.logg.unique()), len(df_line.Z.unique()), 4),
            method='linear'))
    return interpolator_list

def pickling_run():
    from time import perf_counter
    sys.stderr = sys.stdout = open('log.txt', 'w')
    path = '/home/sujays/github/blase/experiments/08_blase3D_HPC_test/emulator_states'
    df = read_state_dicts(path)
    df_gp = df[['teff', 'logg', 'Z']]
    df = df.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    start = perf_counter()
    interpolators = create_interpolators(df, df_gp)
    dump(interpolators, open('interpolator_list.pkl', 'wb'))
    print(f'Interpolator partials dumped to pickle ({perf_counter() - start} s).')

def reconstruct1(wl_grid: np.ndarray, point: np.ndarray, interpolator_list: list[RegularGridInterpolator]) -> np.ndarray:
    output = np.vstack([r for interpolator in interpolator_list if (r := interpolator(point).squeeze())[0] > -100])
    state_dict = {
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    return np.nan_to_num(SLE(wl_native=wl_grid, init_state_dict=state_dict, device='cpu').forward().detach().numpy(), nan=1)

def reconstructn(wl_grid: np.ndarray, points: np.ndarray, interpolator_list: list[RegularGridInterpolator]) -> np.ndarray:
    raw_outputs = [[] for _ in points]
    for interpolator in interpolator_list:
        for i, r in enumerate(interpolator(points).squeeze()):
            if r[0] > -100:
                raw_outputs[i].append(r)
    return np.nan_to_num(np.vstack([SLE(wl_native=wl_grid, init_state_dict={'amplitudes': torch.from_numpy(output[:, 0]),
            'sigma_widths': torch.from_numpy(output[:, 1]),
            'gamma_widths': torch.from_numpy(output[:, 2]),
            'lam_centers': torch.from_numpy(output[:, 3])}, device='cpu').forward().detach().numpy() for output in (np.vstack(raw_output) for raw_output in raw_outputs)]), nan=1)


def rms_loss(wl_grid: np.ndarray, data: np.ndarray, interpolator_list: list[RegularGridInterpolator]) -> Callable:
    return lambda point: ((reconstruct1(wl_grid, point, interpolator_list) - data)**2).mean()**0.5

def inference_test():
    interpolator_list = load(open('interpolator_list.pkl', 'rb'))
    start = perf_counter()
    spec = default_clean(PHOENIXSpectrum(teff=5000, logg=4, Z=0, download=True))
    res = gp_minimize(rms_loss(spec.wavelength.value, spec.flux.value, interpolator_list), dimensions=[(2300, 12000), (2, 6), (-0.5, 0)], n_calls=50, n_random_starts=30)
    print(f'Result: {res.x} achieved in {perf_counter() - start} s')


if __name__ == '__main__':
    """sys.stderr = open('log.txt', 'w')
    sys.stdout = open('out.txt', 'w')
    start = perf_counter()
    interpolator_list = load(open('interpolator_list.pkl', 'rb'))
    R = np.random.default_rng()
    teff_samples = np.arange(3000, 11000.1, 1000)
    logg_samples = np.arange(2, 6.1, 2)
    Z_samples = np.arange(-0.5, 0.1, 0.5)
    print(f'Initializations complete in {perf_counter() - start} s')
    for T, G, Z in product(teff_samples, logg_samples, Z_samples):
        spec = default_clean(PHOENIXSpectrum(teff=int(T), logg=G, Z=Z, download=True))
        start = perf_counter()
        point_random = R.uniform([T-500, 2, -0.5], [T+500, 6, 0], (100, 3))
        reconstructions = reconstructn(spec.wavelength.value, point_random, interpolator_list)
        rms_array = np.sqrt(((reconstructions - spec.flux.value)**2).mean(axis=1))
        res = gp_minimize(rms_loss(spec.wavelength.value, spec.flux.value, interpolator_list), dimensions=[(T-500.0, T+500), (2.0, 6), (-0.5, 0)], n_calls=20, x0=[list(array) for array in point_random], y0=list(rms_array), n_initial_points=0, n_jobs=16)
        print(f'{(T, G, Z)} -> {res.x} achieved in {perf_counter() - start} s')"""
    df = read_state_dicts('/home/sujays/github/blase/experiments/08_blase3D_HPC_test/emulator_states')
    df_gp = df[['teff', 'logg', 'Z']]
    df = df.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    optimize_memory(df)
    df_line = df.query('center <= 8545 and center <= 8544', engine='python').merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
    df.sort_values(['teff', 'logg', 'Z'], inplace=True)
    for i, line in enumerate(df.value_counts('center').index):
        if line <= 8545 and line >= 8544:
            print(f'{i}: {line}')
