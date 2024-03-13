import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch

from blase.emulator import SparseLinearEmulator as SLE
from blase.utils import doppler_grid
from collections import defaultdict
from functools import partial
from os import listdir
from pickle import dump, load
from re import split
from scipy.interpolate import griddata
from tqdm import tqdm


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
        interpolator_list.append(partial(griddata, points=(df_line.teff, df_line.logg, df_line.Z), values=df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy()))
    return interpolator_list

def reconstruct(wl_grid: np.ndarray, interpolators: list[partial], point: tuple[int, float, float]) -> np.ndarray:
    output = np.vstack([r for interpolator in interpolators if (r := interpolator(xi=(point[0], point[1], point[2])))[0] != -1000])
    state_dict = {
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    return np.nan_to_num(SLE(wl_native=wl_grid, init_state_dict=state_dict, device="cpu").forward().detach().numpy(), nan=1)

def local_run(p_teff, p_logg, p_Z):
    path = '/home/sujay/data/10K_12.5K_clones'
    df_native = read_state_dicts(path).query('-0.5 <= Z <= 0.5 & 4000 <= teff <= 7000 & 2 <= logg <= 4')
    df_gp = df_native[['teff', 'logg', 'Z']]
    df = df_native.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    interpolator_list = []
    for line in tqdm(df.center.unique()[:2000]):
        df_line = df.query('center == @line')
        df_line = df_line.merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
        interpolator_list.append(partial(griddata, points=(df_line.teff, df_line.logg, df_line.Z), values=df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy(), method='linear'))

    print('Interpolator partials created')
    output = np.vstack([r for interpolator in interpolator_list if (r := interpolator(xi=(p_teff, p_logg, p_Z)))[0] != -1000])
    state_dict = {
        'pre_line_centers': torch.from_numpy(df.center.unique()[:2000].to_numpy(dtype=np.float64)),
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    wl_lo = 8038
    wl_hi = 12849
    grid = doppler_grid(wl_lo, wl_hi)
    emulator1 = np.nan_to_num(SLE(wl_native=grid, init_state_dict=state_dict, device="cpu").forward().detach().numpy(), nan=1)

    df_native = read_state_dicts(path).query('-0.5 <= Z <= 0.5 & 4000 <= teff <= 7000 & 2 <= logg <= 4 & teff != @p_teff & logg != @p_logg & Z != @p_Z')
    df_gp = df_native[['teff', 'logg', 'Z']]
    df = df_native.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    interpolator_list = []
    for line in tqdm(df.center.unique()[:2000]):
        df_line = df.query('center == @line')
        df_line = df_line.merge(df_gp, how='right', on=['teff', 'logg', 'Z']).fillna(-1000)
        interpolator_list.append(partial(griddata, points=(df_line.teff, df_line.logg, df_line.Z), values=df_line[['amp', 'sigma', 'gamma', 'shift_center']].to_numpy(), method='linear'))

    print('Interpolator partials created')
    output = np.vstack([r for interpolator in interpolator_list if (r := interpolator(xi=(p_teff, p_logg, p_Z)))[0] != -1000])
    state_dict = {
        'pre_line_centers': torch.from_numpy(df.center.unique()[:2000].to_numpy(dtype=np.float64)),
        'amplitudes': torch.from_numpy(output[:, 0]),
        'sigma_widths': torch.from_numpy(output[:, 1]),
        'gamma_widths': torch.from_numpy(output[:, 2]),
        'lam_centers': torch.from_numpy(output[:, 3]),
    }
    emulator2 = np.nan_to_num(SLE(wl_native=grid, init_state_dict=state_dict, device="cpu").forward().detach().numpy(), nan=1)
    pd.DataFrame({'wavelength': grid, 'flux1': emulator1, 'flux2': emulator2}).to_parquet('plot_residual_hist.parquet.gz', compression='gzip')

def triton_run():
    sys.stderr = open('log', 'w')
    path = '/home/sujays/github/blase/experiments/08_blase3D_HPC_test/emulator_states'
    df = read_state_dicts(path)
    df_gp = df[['teff', 'logg', 'Z']]
    df = df.explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    interpolators = create_interpolators(df, df_gp)
    dump(interpolators, open('interpolators.pkl', 'wb'))
    with open('log.txt') as f:
        f.write('Interpolator partials dumped to pickle.')

if __name__ == '__main__':
    #local_run(4100, 2.5, 0.0)
    triton_run()
