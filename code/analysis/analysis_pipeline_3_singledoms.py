import os
import numpy as np
from nilearn import image
from scipy.stats import false_discovery_control as fdr
from scipy.stats import ttest_ind

def ttest_domains(results, dom_combs, dom_combs_str, fdr_opt=True, save=False):
    
    # Initialize results matrices
    ts = np.empty((results.shape[1], len(dom_combs)))
    pvals = np.empty((results.shape[1], len(dom_combs)))

    # Run t-test for each ROI and each domains pairing
    for roi in range(results.shape[1]):  
        for comb in dom_combs:
            d1 = results[:, roi, comb[0]]
            d2 = results[:, roi, comb[1]]
            t, pval = ttest_ind(d1, d2)
            ts[roi,comb] = t 
            pvals[roi,comb] = pval
    
    # FDR correction for multiple comparisons
    if fdr_opt:
        pvals = np.reshape(fdr(np.ravel(pvals)), pvals.shape)

    # Save results
    if save:
        np.savez('/data1/Action_teresi/CCA/ttest_cca', tstat=ts, pvals=pvals, domains_contrasts=dom_combs_str, FDRcorrection=fdr_opt)

    return ts, pvals

def save_nifti(results, briks, atlas_file='Schaefer200', savepath='', ROIsmask=[]):
    
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

    if len(ROIsmask) <= 0:
        ROIsmask = atlas_rois

    x,y,z = atlas.get_fdata().shape
    
    image_final = np.squeeze(np.zeros((x,y,z,briks)))

    for r, roi in enumerate(ROIsmask):
        x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
        
        image_final[x_inds, y_inds, z_inds] = results[:, r]

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename(savepath)

    return

if __name__ == '__main__': 
    
    # Set options and parameters
    condition = 'aud'
    save = True
    save_nifti_opt = True
    atlas_file = 'Schaefer200'
    alpha = 0.05
    maxT = True

    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'

    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']

    # Filter ROIs with significant R2 for full model
    pvals_group_fm = np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals{}_group_{}.npz'.format(global_path, condition, '_maxT' if maxT else '', atlas_file), allow_pickle=True)['pvals_group']
    ROIs_sign_idx = np.unique(np.where(np.squeeze(pvals_group_fm) < alpha)[0])
    ROIs_sign = np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals{}_group_{}.npz'.format(global_path, condition, '_maxT' if maxT else '', atlas_file), allow_pickle=True)['rois_list'][ROIs_sign_idx]

    # Save list of significant ROIs
    np.savetxt('{}cca_results/{}/group/fullmodel/significantROIs_{}.txt'.format(global_path, condition, condition), ROIs_sign)

    # Load R2 from full model and variance partitioning, select first perm (true R), and significative ROIs 
    results_vp = np.load('{}cca_results/{}/group/variancepart/CCA_R2_allsubs_{}.npz'.format(global_path, condition, atlas_file), allow_pickle=True)['results'][:, ROIs_sign_idx, 0, :]
    results_fm = np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals_allsubs_{}.npz'.format(global_path, condition, atlas_file), allow_pickle=True)['results'][:, ROIs_sign_idx, 0, :]
   
    # Get R2 for single domain (subtract shuffled models R2 from full model R2)
    res_dom = np.squeeze(results_fm - results_vp)

    # Get mean across subs and concatenate to main matrix
    res_dom_mean = np.mean(res_dom, axis=0)
    res_dom_final = np.concatenate((res_dom, np.expand_dims(res_dom_mean, axis=0)), axis=0)

    # Save results
    if save:
        path = '{}/cca_results/{}/group/single_doms/'.format(global_path, condition)
        if not os.path.exists(path):
            os.makedirs(path)

        np.savez('{}CCA_R2_group_singledoms'.format(path), results_group_sd=res_dom_mean, rois_list=ROIs_sign)
        np.savez('{}CCA_R2_allsubs_singledoms'.format(path), results_subs_sd=res_dom, rois_list=ROIs_sign)
    
    # Save nifti
    if save_nifti_opt:
        for dom in range(res_dom.shape[-1]):
            save_nifti(res_dom_final[:,:,dom], res_dom_final.shape[0], atlas_file, '{}CCA_R2_{}_{}.nii'.format(path, domains_list[dom], atlas_file), ROIs_sign)