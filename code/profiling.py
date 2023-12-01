import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from check_collinearity import canoncorrelation
from canon_corr_fmri import permutation_schema, run_canoncorr, gen_fmri_signal, gen_correlated_data
import numpy as np
import time
import multiprocessing as mp

if __name__ == '__main__':

    t = time.time()

    # Set parameters
    n_rois = 30 #400
    n_voxels = 100
    n_perms = 500 #1000
    n_tpoints = 1614
    chunk_size = 15 # add check max action duration
    seed = 0

    
    # Set data_path as the path where the csv files containing single domain matrices are saved, including first part of filename, up to the domain specification (here I specify 'tagging_carica101_group_2su3_convolved_' for example)
    data_path = 'Data/Carica101_Models/Domains/tagging_carica101_group_2su3_convolved_'

    # Set out_path as the path where to save results figures
    out_path = 'Results/Collinearity/New/'

    # Load Models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt(data_path + '{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}

    # Generate fake correlated data starting from domain
    data = {r: gen_correlated_data(domains['space_movement'], n_voxels, noise=2) for r in range(n_rois)}

    # Generate fake simulated data (simple oscillations)
    #data = {r: gen_fmri_signal() for r in range(n_rois)}
    
    # Generate permutation schema
    perm_schema = permutation_schema(n_tpoints, n_perms=n_perms, chunk_size=chunk_size)

    # Initialize results matrix
    result_matrix = np.full((n_rois, n_perms+1, len(domains)), np.nan)

    #### Get R2 With pool
    t1 = time.time()
    results_pool = []

    pool = mp.Pool(15)

    for r, roi in data.items():
        result_pool = pool.apply_async(run_canoncorr, args=(roi, perm_schema, domains, True))
        results_pool.append(result_pool)
    
    pool.close()
        

    for result_pool in results_pool:
        njob = result_pool._job
        result_matrix[njob, :, :] = result_pool.get()
            
    np.save('results_n2', result_matrix)
    
    print(time.time() - t1)
    print('d')