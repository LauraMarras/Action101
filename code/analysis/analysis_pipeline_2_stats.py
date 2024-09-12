import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from nilearn import image
from stats_funcs import get_pvals_sub, get_pvals_group, save_nifti, get_results_sub
from scipy.stats import false_discovery_control as fdr

if __name__ == '__main__': 
    
    # Set options and parameters
    condition = 'vid'
    full_model_opt = True # full_model vs variance_partitioning (if False run Variance partitioning)

    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    n_perms = 1000 if full_model_opt else 0
    atlas_file = 'Schaefer200'
    rois_to_include = list(np.loadtxt('{}cca_results/AV/group/fullmodel/significantROIs_AV.txt'.format(global_path)).astype(int)) if condition != 'AV' else []
    suffix = 'fullmodel' if full_model_opt else 'variancepart'

    ss_stats = False
    adjusted = False

    group_stats = True
    maxT = True
    FDR = False

    save = True
    save_stats_nifti = False

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

    # Single subject stats
    if ss_stats:
        print('Starting single-sub statistical analyses')

        # Get pvalues for all subjects
        rois_list = np.full((len(sub_list), n_rois), np.nan)
        results_subs = np.full((len(sub_list), n_rois, n_perms+1, n_doms), np.nan)

        if n_perms>0:
            pvals_subs = np.full((len(sub_list), n_rois, n_perms+1, n_doms), np.nan)

            for s, sub in enumerate(sub_list):
                sub_str = str(sub) if len(str(sub))>=2 else '0'+str(sub)
                results_subs[s], pvals_subs[s], rois_list[s] = get_pvals_sub(condition, sub_str, adjusted=adjusted, save=save, suffix=suffix, atlas_file=atlas_file, global_path=global_path, save_nifti_opt=save_stats_nifti)
                        
            # Verify that ROIs list is the same for all subjects (i.e. that no ROI is absent in any subject)
            if np.any(np.diff(rois_list, axis=0)):
                print('ROI list is not the same in every subject, check ROIs singularly')
            
            else:
                rois_list = rois_list[0].astype(int)
            
            # Save results and uncorrected pvals to numpy
            path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            np.savez('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), results=results_subs, pvals=pvals_subs, rois_list=rois_list)

        else:
            for s, sub in enumerate(sub_list):
                sub_str = str(sub) if len(str(sub))>=2 else '0'+str(sub)
                results_subs[s], rois_list[s] = get_results_sub(condition, sub_str, adjusted=adjusted, save=save, suffix=suffix, atlas_file=atlas_file, global_path=global_path, save_nifti_opt=save_stats_nifti)
            
             # Verify that ROIs list is the same for all subjects (i.e. that no ROI is absent in any subject)
            if np.any(np.diff(rois_list, axis=0)):
                print('ROI list is not the same in every subject, check ROIs singularly')
            
            else:
                rois_list = rois_list[0].astype(int)
            
            # Save results and uncorrected pvals to numpy
            path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            np.savez('{}CCA_R2{}_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), results=results_subs, rois_list=rois_list)

        print('Finished single-sub statistical analyses')


    # Group level stats
    if group_stats:
        print('Starting group statistical analyses')
        
        # Load single subject results
        path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
        results = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['results']
        rois = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['rois_list']
        
        if n_perms > 0:
            # Load single subject pvals
            pvals = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['pvals']
        
            # Get aggregated results and pvals
            results_group, pvals_group = get_pvals_group(condition, rois, pvals, results, maxT=maxT, FDR=FDR, save=save, suffix=suffix, global_path=global_path, save_nifti_opt=save_stats_nifti)
        
        print('Finished group statistical analyses')
