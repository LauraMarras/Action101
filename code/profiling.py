import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from check_collinearity import canoncorrelation
from canon_corr_fmri import permutation_schema, run_canoncorr, gen_fmri_signal, gen_correlated_data
import numpy as np
from simulation.utils.exectime_decor import timeit
import multiprocessing as mp
from nilearn import image
import sys


def extract_roi(data, atlas):
        
    rois = np.unique(atlas)
    rois = np.delete(rois, np.argwhere(rois==0))
    n_rois = len(rois)

    data_rois = {}
    rois_nvoxels = {}
    
    for roi in rois:
        (x,y,z) = np.where(atlas==roi)
        if data.ndim == 4:
            data_roi = data[x,y,z,:]

        elif data.ndim ==3:
            data_roi = data[x,y,z]

        
        data_rois[int(roi)] = data_roi.T
        rois_nvoxels[int(roi)] = len(x)

    return data_rois, n_rois, rois_nvoxels

@timeit
def call_cca_rois(data_rois, n_rois, n_perms, domains, perm_schema, adjusted, pooln=20):
    
    # Initialize results matrix
    result_matrix = np.full((n_rois, n_perms+1, len(domains)), np.nan)
    result_dict = {}

    result_matrix_adj = np.full((n_rois, n_perms+1, len(domains)), np.nan)
    result_dict_adj = {}

    #### Get R2 With pool
    results_pool = []

    pool = mp.Pool(pooln)

    for r, roi in data_rois.items():
        result_pool = pool.apply_async(run_canoncorr, args=(roi, perm_schema, domains, adjusted))
        results_pool.append((r, result_pool))

    pool.close()
        
    for result_pool in results_pool:
        njob = result_pool[1]._job
        roi_n = result_pool[0]
        result_matrix[njob, :, :] = result_pool[1].get()[0]
        result_matrix_adj[njob, :, :] = result_pool[1].get()[1]
        result_dict_adj[roi_n] = result_pool[1].get()[1]
        result_dict[roi_n] = result_pool[1].get()[0]

    return result_matrix, result_dict, result_matrix_adj, result_dict_adj


if __name__ == '__main__':
    
    res = np.load('data/cca_results/sub-1/res_sub-1_adj_200.npy')
    resna = np.load('data/cca_results/sub-1/res_sub-1_200.npy')

    
    print('starting')

    # Set parameters
    sub=1
    n_perms = 2 #1000
    chunk_size = 15 # add check max action duration
    seed = 0
    atlas_file = 'atlas_2orig'
    fsuffix = '_200' if atlas_file == 'atlas_2orig' else '_1000'

    # Print output to txt file
    log_path = 'data/cca_logs/sub-{}/'.format(sub)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    orig_stdout = sys.stdout
    logfile = open('{}logs_sub-{}{}.txt'.format(log_path, sub, fsuffix), 'w')
    sys.stdout = logfile
    
    # Set model_path as the path where the csv files containing single domain matrices are saved, including first part of filename, up to the domain specification (here I specify 'tagging_carica101_group_2su3_convolved_' for example)
    model_path = 'data/models/domains/group_ds_conv_'

    # Set out_path as the path where to save results figures
    out_path = 'Results/Collinearity/New/'

    # Load Models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt(model_path + '{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}

    # Load Data
    data = image.load_img('data/simulazione_preprocessed/sub-{}/func/cleaned.nii.gz'.format(sub)).get_fdata()
    atlas = image.load_img('data/simulazione_datasets/sub-{}/anat/{}.nii.gz'.format(sub, atlas_file)).get_fdata()
    
    # Extract rois
    data_rois, n_rois, n_voxles = extract_roi(data, atlas)

    n_tpoints = data.shape[-1]

    # Generate permutation schema
    perm_schema = permutation_schema(n_tpoints, n_perms=n_perms, chunk_size=chunk_size)

    # Run cca for each roi
    result_matrix, result_dict, result_matrix_adj, result_dict_adj = call_cca_rois(data_rois, n_rois, n_perms, domains, perm_schema, adjusted=True, pooln=10)
    
    # Save
    folder_path = 'data/cca_results/sub-{}/'.format(sub)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save('{}res_sub-{}{}'.format(folder_path, sub, fsuffix), result_matrix)
    np.save('{}res_sub-{}_adj{}'.format(folder_path, sub, fsuffix), result_matrix_adj)
    
    #print('time to run cca for each roi, with {} permutations:       '.format(n_perms), (time.time() - t1))

    # Close logfile
    sys.stdout = orig_stdout
    logfile.close()

    print('finished')