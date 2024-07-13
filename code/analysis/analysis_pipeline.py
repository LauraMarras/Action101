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
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    n_subs = len(sub_list)
    
    cca = False
    n_perms = 1000
    chunk_size = 15
    seed = 0
    atlas_file = 'Schaefer200'
    pooln = 20
    suffix = '_pca'

    ss_stats = False
    save = True
    run_fdr = True
    group_stats = True
    maxT = False

    # Load task models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    # Load Atlas
    atlas = image.load_img('/home/laura.marras/Documents/Atlases/{}.nii.gz'.format(atlas_file))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

    # CCA
    if cca:
        print('Starting CCA')
        
        # Run CCA for all subjects
        run_cca_all_subjects(sub_list, domains, atlas_file, n_perms, chunk_size, seed, pooln, save, suffix)

        print('Finished CCA')

    # Single subject stats
    if ss_stats:
        print('Starting single-sub statistical analyses')

        # Get pvalues for all subjects
        results_subs = np.full((len(sub_list), len(atlas_rois), n_perms+1, n_doms), np.nan)
        pvals_subs = np.full((len(sub_list), len(atlas_rois), n_perms+1, n_doms), np.nan)

        for s, sub in enumerate(sub_list):

            results_subs[s], pvals_subs[s] = get_pvals_sub(sub, save=save, suffix=suffix)
                        
            # Build dictionaries
            res_dict = {r+1: results_subs[s,r,0,:] for r in range(len(atlas_rois))}
            pvals_dict = {r+1: pvals_subs[s,r,0,:] for r in range(len(atlas_rois))}

            # FDR correction
            if run_fdr:
                pvals_array = np.ravel(pvals_subs[s,:,0,:], order='F')
                pvals_subs_fdr = fdr(pvals_array)
                
                # Rebuild pvals_dictionary
                pvals_dict = {roi: pvals_array[r*n_doms:r*n_doms+n_doms] for r, roi in enumerate(atlas_rois)}

            # Save as nifti 
            if save:
                folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/CCA_res_sub-{}_{}{}.nii'.format(sub, suffix, sub, atlas_file, '_fdr' if run_fdr else '')
                image_final = save_nifti(atlas, n_doms, res_dict, pvals_dict, folder_path)

        # Save results and uncorrected pvals to numpy
        np.savez('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs{}_{}.npz'.format(suffix, atlas_file), results=results_subs, pvals=pvals_subs)

        print('Finished single-sub statistical analyses')


    # Group level stats
    if group_stats:
        print('Starting group statistical analyses')
        
        # Load single subject results and pvals
        results = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['results']
        pvals = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['pvals']
        
        # Get aggregated results and pvals
        results_group, pvals_group = get_pvals_group(atlas_rois, pvals, results, maxT=maxT, save=save, suffix='{}_{}'.format(suffix, atlas_file))
        correction =  '_maxT' if maxT else ''
        
        # Build dictionaries
        res_dict = {roi: results_group[r,:] for r, roi in enumerate(atlas_rois)}
        pvals_dict = {roi: pvals_group[r,:] for r, roi in enumerate(atlas_rois)}

        # Correct for multiple comparisons FDR
        if run_fdr:
            correction += '_fdr'
            pvals_array = np.ravel(pvals_group, order='F')
            pvals_fdr = fdr(pvals_array)

            # Rebuild pvals_dictionary
            pvals_dict = {roi: pvals_array[r*n_doms:r*n_doms+n_doms] for r, roi in enumerate(atlas_rois)}

        # Save as nifti
        folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}{}.nii'.format(suffix, atlas_file, correction) 
        image_final = save_nifti(atlas, n_doms, res_dict, pvals_dict, folder_path)

        print('Finished group statistical analyses')
