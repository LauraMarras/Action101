import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from nilearn import image
from canonical_correlation_funcs import run_cca_all_subjects, pca_all_rois
from stats_funcs import get_pvals_sub, get_pvals_group, save_nifti
from scipy.stats import false_discovery_control as fdr

if __name__ == '__main__': 

    # Set parameters
    sub_list = np.array([18, 19, 22, 32])
    n_subs = len(sub_list)
    
    cca = True
    n_perms = 1000
    chunk_size = 15
    seed = 0
    atlas_file = 'Schaefer200'
    pooln = 20
    suffix = '_pca'

    stats = True
    save = True
    run_fdr = False
    group_stats = False

    # Load task models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    # CCA
    if cca:
        print('Starting CCA')
        
        # Run CCA for all subjects
        run_cca_all_subjects(sub_list, domains, atlas_file, n_perms, chunk_size, seed, pooln, save, suffix)

        print('Finished CCA')

    # Single subject stats
    if stats:
        print('Starting single-sub statistical analyses')

        # Load Atlas
        atlas = image.load_img('/home/laura.marras/Documents/Atlases/{}.nii.gz'.format(atlas_file))
        atlas_rois = np.unique(atlas.get_fdata()).astype(int)
        atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

        # Get pvalues for all subjects
        results_subs = {}
        pvals_subs = {}
        for s, sub in enumerate(sub_list):

            results_subs[sub], pvals_subs[sub] = get_pvals_sub(sub, save=True, suffix=suffix)
            res_dict = {k: v[0,:] for k,v in sorted(results_subs[sub].items())}
            pvals_dict = {k: v[0,:] for k,v in sorted( pvals_subs[sub].items())}

            
            # FDR correction
            if run_fdr:
                pvals_array = np.concatenate([v[0,:] for k,v in sorted(pvals_subs[sub].items())], 0)
                pvals_fdr = fdr(pvals_array)

                # Rebuild pvals_dictionary
                pvals_fdr_dict = {roi: pvals_array[r*n_doms:r*n_doms+n_doms] for r, roi in enumerate(atlas_rois)}

                # Save as nifti
                folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/CCA_res_sub-{}_{}_fdr.nii'.format(sub,suffix, sub, atlas_file)
                image_final = save_nifti(atlas, n_doms, res_dict, pvals_fdr_dict, folder_path)
            
            else:
                # Save as nifti
                folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/CCA_res_sub-{}_{}.nii'.format(sub,suffix, sub, atlas_file)
                image_final = save_nifti(atlas, n_doms, res_dict, pvals_dict, folder_path)

        print('Finished single-sub statistical analyses')


    # Group results
    if group_stats:
        print('Starting group statistical analyses')

        results_group, pvals_group = get_pvals_group(atlas_rois, pvals_subs, results_subs, n_perms+1, n_doms, save=True)

        # Correct for multiple comparisons FDR
        pvals_array = np.concatenate([v for k,v in sorted(pvals_group.items())], 0)
        pvals_fdr = fdr(pvals_array)

        # Rebuild pvals_dictionary
        pvals_fdr_group_dict = {roi: pvals_array[r*n_doms:r*n_doms+n_doms] for r, roi in enumerate(atlas_rois)}

        # Save as nifti
        folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/cca_res_group_fdr.nii'
        image_final = save_nifti(atlas, n_doms, results_group, pvals_fdr_group_dict, folder_path)

        print('Finished group statistical analyses')
