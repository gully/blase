import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from functools import partial
from os import listdir
from re import split
from scipy.interpolate import griddata
from tqdm import tqdm


def read_state_dicts(path: str) -> pd.DataFrame:
    line_stats = defaultdict(list)
    for f in tqdm(listdir(path)):
        state_dict = state_dict = torch.load(f'{path}/{f}', map_location='cuda:0')
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

def local_run():
    path = '/home/sujay/data/10K_12.5K_clones'
    df = read_state_dicts(path).query('Z == 0').explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')
    interpolator_list = []
    for line in tqdm(df.center.unique()):
        df_line = df.query('center == @line')
        interpolator_list.append(partial(griddata, points=(df_line.teff, df_line.logg), values=df_line[['amp', 'sigma', 'gamma']].to_numpy(), method='linear'))
    print('Interpolator partials created')
    for interpolator in interpolator_list:
        print(interpolator(xi=(5777, 4.44))) 

def triton_run():
    path = '/home/sujays/github/blase/experiments/08_blase3D_HPC_test/emulator_states'
    df = read_state_dicts(path).explode(['center', 'amp', 'sigma', 'gamma', 'shift_center']).convert_dtypes(dtype_backend='numpy_nullable')
    print('DataFrame created')
    optimize_memory(df)
    print('DataFrame memory optimized')

if __name__ == '__main__':
    local_run()
    #triton_run()
