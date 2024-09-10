import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from nilearn import image
from canonical_correlation_funcs import run_cca_all_subjects

if __name__ == '__main__': 
    
    # Set options
    cca = False
    condition = 'vid'
    full_model_opt = True # full_model vs variance_partitioning (if False run Variance partitioning)
    pooln = 16

    # Set parameters
    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    n_perms = 1000 if full_model_opt else 0
    chunk_size = 15
    seed = 0
    atlas_file = 'Schaefer200'
    rois_to_include = list(np.loadtxt('/data1/Action_teresi/CCA/cca_results/group/significantROIs_AV.txt').astype(int)) if condition != 'AV' else []
    zscore_opt = False
    skip_roi = False
    variance_part = 0 if full_model_opt else 50
    suffix = 'fullmodel' if full_model_opt else 'variancepart'
    save = True

    # Load task and Create Full model
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    full_model = {'full_model': np.hstack([domains[d] for d in domains_list])}
    if full_model_opt:
        domains = full_model

    n_doms = len(domains.keys())

    # Load Atlas
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    x,y,z = atlas.get_fdata().shape

    # If indicated, select only specified ROIs 
    if len(rois_to_include)<=0:
        rois_to_include = atlas_rois
    
    rois = np.array([r for r in rois_to_include if r in atlas_rois])
    n_rois = len(rois)

    # CCA
    if cca:
        print('Starting CCA')
        
        # Run CCA for all subjects
        run_cca_all_subjects(condition, sub_list, domains, atlas_file, rois_to_include, n_perms, chunk_size, seed, pooln, zscore_opt, skip_roi, variance_part, save, suffix)

        print('Finished CCA')