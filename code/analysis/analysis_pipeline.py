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
from permutation_schema_func import permutation_schema

if __name__ == '__main__': 

    # Set parameters
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    cca = False
    full_model_opt = True
    n_perms = 1000
    chunk_size = 15
    seed = 0
    atlas_file = 'Schaefer200'
    pooln = 25
    zscore_opt = False
    skip_roi = False
    variance_part = 0
    suffix = '_pcanoz_fullmodel_6doms' #'_pcanoz_fullmodel' #'_pca_variancepart' # '_pca_fullmodel' #

    ss_stats = False
    adjusted = False
    save = True
    run_fdr = False

    group_stats = False
    maxT = True

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

    # CCA
    if cca:
        print('Starting CCA')
        
        # Run CCA for all subjects
        run_cca_all_subjects(sub_list, domains, atlas_file, n_perms, chunk_size, seed, pooln, zscore_opt, skip_roi, variance_part, save, suffix)

        print('Finished CCA')

    # Single subject stats
    if ss_stats:
        print('Starting single-sub statistical analyses')

        # Get pvalues for all subjects
        results_subs = np.full((len(sub_list), len(atlas_rois), n_perms+1, n_doms), np.nan)
        pvals_subs = np.full((len(sub_list), len(atlas_rois), n_perms+1, n_doms), np.nan)

        for s, sub in enumerate(sub_list):

            results_subs[s], pvals_subs[s] = get_pvals_sub(sub, adjusted=adjusted, save=save, suffix=suffix, atlas_file=atlas_file, global_path=global_path)
                        
            # Build dictionaries
            res_dict = {r+1: results_subs[s,r,0,:].squeeze() for r in range(len(atlas_rois))}
            pvals_dict = {r+1: pvals_subs[s,r,0,:].squeeze() for r in range(len(atlas_rois))}

            # FDR correction
            if run_fdr:
                pvals_array = np.ravel(pvals_subs[s,:,0,:].squeeze(), order='F')
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
        
        # Load single subject results
        results = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['results']
        
        if n_perms > 0:
            # Load single subject pvals
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

            # Save numpy
            np.savez('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}{}'.format(suffix, atlas_file, correction), results_group=results_group, pvals_group=pvals_group)

            # Save as nifti
            folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}{}.nii'.format(suffix, atlas_file, correction) 
            image_final = save_nifti(atlas, n_doms, res_dict, pvals_dict, folder_path)

        else:
            # Get group results by average across subs
            results_group = np.mean(results, axis=0).squeeze()
            
            if save:
                path = global_path + 'cca_results/group/'
                if not os.path.exists(path):
                    os.makedirs(path)
                
                # Save numpy
                np.savez(path + 'CCA_res_group{}_{}'.format(suffix, atlas_file), results_group=results_group)  
                
                # Create nifti
                image_final = np.squeeze(np.zeros((x,y,z,n_doms)))

                for r, roi in enumerate(atlas_rois):
                    x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
                    
                    image_final[x_inds, y_inds, z_inds] = results_group[r]

                img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
                img.to_filename('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}.nii'.format(suffix, atlas_file))
                
        print('Finished group statistical analyses')
