import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from nilearn import image
from canonical_correlation_funcs import run_cca_all_subjects
from stats_funcs import get_pvals_sub, get_pvals_group, save_nifti

if __name__ == '__main__': 

    print('Starting CCA')

    # Set parameters
    sub_list = np.array([7,8,9])
    n_subs = len(sub_list)
    save = True

    ## CCA
    atlas_file = 'atlas_2orig' # 'atlas1000_2orig.nii.gz'
    pooln = 5

    ## Permutation schema
    n_perms = 1000
    chunk_size = 15
    seed = 0
    
    # Load task models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    # Run CCA for all subjects
    run_cca_all_subjects(sub_list, domains, atlas_file, n_perms, chunk_size, seed, pooln, save)

    print('Finished CCA')

    print('Starting statistical analyses')
    
    # Load Atlas
    atlas = image.load_img('/home/laura.marras/Documents/Atlases/Schaefer-200_7Networks_ICBM152_Allin.nii.gz')
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    
    # Get pvalues for all subjects
    results_subs = {}
    pvals_subs = {}
    for s, sub in enumerate(sub_list):
        results_subs[sub], pvals_subs[sub] = get_pvals_sub(sub, save=True)

    # Get group results
    results_group, pvals_group = get_pvals_group(atlas_rois, pvals_subs, results_subs, n_perms+1, n_doms, save=True)

    # Save as nifti
    folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/debug/'
    image_final = save_nifti(atlas, n_doms, results_group, pvals_group, folder_path)

    print('Finished statistical analyses')