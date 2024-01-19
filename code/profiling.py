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
from nilearn import image
import sys

def extract_roi(data, atlas):
        
    rois = np.unique(atlas)
    rois = np.delete(rois, np.argwhere(rois==0))
    n_rois = len(rois)

    data_rois = {}

    for roi in rois:
        (x,y,z) = np.where(atlas==roi)
        data_roi = data[x,y,z,:]
        data_rois[int(roi)] = data_roi.T
   

    return data_rois, n_rois



if __name__ == '__main__':

    t = time.time()

    # Set parameters
    sub=1
    n_perms = 10 #1000
    chunk_size = 15 # add check max action duration
    seed = 0

    # Print output to txt file
    orig_stdout = sys.stdout
    logfile = open('data/results/logs_sub-0{}.txt'.format(sub+1), 'w')
    sys.stdout = logfile
    
    # Set model_path as the path where the csv files containing single domain matrices are saved, including first part of filename, up to the domain specification (here I specify 'tagging_carica101_group_2su3_convolved_' for example)
    model_path = 'data/models/Domains/group_ds_conv_'

    # Set out_path as the path where to save results figures
    out_path = 'Results/Collinearity/New/'

    # Load Models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt(model_path + '{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}

    # Load Data
    data = image.load_img('data/simulazione_results/sub-0{}/func/cleaned.nii.gz'.format(sub+1)).get_fdata()
    atlas = image.load_img('data/simulazione_datasets/sub-0{}/atlas_2orig.nii.gz'.format(sub+1)).get_fdata()

    # Extract rois
    data_rois, n_rois = extract_roi(data, atlas)

    n_tpoints = data.shape[-1]

    # Generate permutation schema
    perm_schema = permutation_schema(n_tpoints, n_perms=n_perms, chunk_size=chunk_size)

    # Initialize results matrix
    result_matrix = np.full((n_rois, n_perms+1, len(domains)), np.nan)

    #### Get R2 With pool
    t1 = time.time()
    results_pool = []

    pool = mp.Pool(20)

    for r, roi in data_rois.items():
        result_pool = pool.apply_async(run_canoncorr, args=(roi, perm_schema, domains, True))
        results_pool.append(result_pool)
    
    pool.close()
        

    for result_pool in results_pool:
        njob = result_pool._job
        result_matrix[njob, :, :] = result_pool.get()
            
    np.save('data/results/results_sub-0{}'.format(sub+1), result_matrix)
    
    print('time to run cca for each roi, with {} permutations:       '.format(n_perms), (time.time() - t1))

    # Close logfile
    sys.stdout = orig_stdout
    logfile.close()