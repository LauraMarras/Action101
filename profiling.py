if __name__ == '__main__':

    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['BLIS_NUM_THREADS'] = '1'

    #from check_collinearity import canoncorrelation
    from canon_corr_fmri import generate_signal, permutation_schema, run_canoncorr
    import numpy as np
    import time
    import multiprocessing as mp
    
    t = time.time()

    # Set parameters
    n_rois = 10#400
    n_voxels = 100
    n_perms = 10 #1000
    n_tpoints = 1614
    chunk_size = 15
    seed = 0

    # Generate simulated data (to be replaced with loading of real data)
    data = {r: generate_signal() for r in range(n_rois)}

    # Set data_path as the path where the csv files containing single domain matrices are saved, including first part of filename, up to the domain specification (here I specify 'tagging_carica101_group_2su3_convolved_' for example)
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/Domains/tagging_carica101_group_2su3_convolved_'

    # Set out_path as the path where to save results figures
    out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Results/Collinearity/New/'

    # Load Models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt(data_path + '{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}

    # Generate permutation schema
    perm_schema = permutation_schema(n_tpoints, n_perms=n_perms)

    # Initialize results matrix
    results = np.full((n_rois, n_perms+1, len(domains)), np.nan)

    #### Get R2 With pool
    t1 = time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        t2 = time.time() - t1
        
        results = np.array([pool.apply(run_canoncorr, args=(roi, data, perm_schema, domains)) for roi in range(n_rois)])
        
        t3  = time.time() - t1
        
        for roi in range(n_rois):
            results[roi, :, :] = run_canoncorr(roi, data, perm_schema, domains)
        
        t4 = time.time() - t1
    
    #### Without pool
    t5 = time.time() - t1

    for roi in range(n_rois):
            results[roi, :, :] = run_canoncorr(roi, data, perm_schema, domains)
    
    t6 = time.time() - t1

    print('time before pool  ', t1 - t)
    print('time to call pool  ', t2)
    print('time to run with pool.apply  ', t3 - t2)
    print('time to run with pool for loop  ', t4 - t3)
    print('time to finish pool  ', t5 - t4)
    print('time to run without for loop  ', t6 - t5)