import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from functools import reduce
from numpy.polynomial.polynomial import polyvander2d, polyvander3d
from os import listdir
from re import split
from scipy.linalg import lstsq
from tqdm import tqdm

def main():
    path = "/home/sujay/data/10K_12.5K_clones"

    line_stats = defaultdict(list)
    for state_file in tqdm(listdir(path)):
        state_dict = torch.load(f'{path}/{state_file}', map_location='cuda:0')
        tokens = split('[TGZ]', state_file[:-3])
        line_stats['teff'].append(int(tokens[1]))
        line_stats['logg'].append(float(tokens[2]))
        line_stats['Z'].append(float(tokens[3]))

        line_stats['center'].append(state_dict['pre_line_centers'].cpu().numpy())
        line_stats['shift_center'].append(state_dict['lam_centers'].cpu().numpy())
        line_stats['amp'].append(state_dict['amplitudes'].cpu().numpy())
        line_stats['sigma'].append(state_dict['sigma_widths'].cpu().numpy())
        line_stats['gamma'].append(state_dict['gamma_widths'].cpu().numpy())

    line_set = reduce(np.union1d, line_stats['center'])
    print('Creating DataFrame...')
    df = pd.DataFrame(line_stats).query('Z == 0').explode(['center', 'amp', 'sigma', 'gamma', 'shift_center'])
    df = df.convert_dtypes(dtype_backend='numpy_nullable')

    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns

    print('Optimizing DataFrame Memory Usage...')
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    line_counts = df.value_counts('center')

    current_line = line_counts.index[0]
    MAX_DEGREE = 3
    df_line = df.query('center == @current_line')
    '''
    print('Creating Design and Observation Matrices...')
    design_matrix = polyvander3d(df_line['amp'], df_line['sigma'], df_line['gamma'], deg=(MAX_DEGREE, MAX_DEGREE, MAX_DEGREE)).astype(float)
    observation_matrix = df_line[['teff', 'logg', 'Z']].to_numpy(dtype=float)
    print('Solving for Least Squares Coefficients...')
    coefficients, residuals = lstsq(design_matrix, observation_matrix)[0:2]
    print('Evaluating Akaike Information Criterion...')
    loss_function = np.sum(residuals**2)
    AIC = 2 * (coefficients.size - np.log(loss_function))
    # Use inflect ordinals in the future, because why not
    print(f'{MAX_DEGREE}-th Order Raw Model AIC: {AIC}')
    # Loop over design matrix, and test drop columns to improve AIC
    # Refined Model AIC here
    manifold = design_matrix @ coefficients
    df_visual = pd.DataFrame({'amp': df_line['amp'], 'sigma': df_line['sigma'], 'gamma': df_line['gamma'], 'teff': manifold[:, 0], 'logg': manifold[:, 1], 'Z': manifold[:, 2]})
    print('Writing Manifold to Parquet...')
    df_visual.to_parquet('experiments/09_surface_fitting/surface_visualization_sparse.parquet.gz', compression='gzip')'''

    print('Creating Design and Observation Matrices...')
    design_matrix = polyvander2d(df_line['teff'], df_line['logg'], deg=(7, 7)).astype(float)
    observation_matrix = df_line[['amp', 'sigma', 'gamma']].to_numpy(dtype=float)
    print('Solving for Least Squares Coefficients...')
    coefficients, residuals = lstsq(design_matrix, observation_matrix)[0:2]
    print('Evaluating Akaike Information Criterion...')
    loss_function = np.sum(residuals**2)
    AIC = 2 * (coefficients.size - np.log(loss_function))
    # Use inflect ordinals in the future, because why not
    print(f'{MAX_DEGREE}-th Order Raw Model AIC: {AIC}')
    print(f'Design Matrix Shape: {design_matrix.shape}')
    manifold = design_matrix @ coefficients
    df_visual = pd.DataFrame({'teff': df_line['teff'], 'logg': df_line['logg'], 'amp': manifold[:, 0], 'sigma': manifold[:, 1], 'gamma': manifold[:, 2]})
    df_visual.to_parquet('experiments/09_surface_fitting/surface_visualization_grid.parquet.gz', compression='gzip')

if __name__ == '__main__':
    main()