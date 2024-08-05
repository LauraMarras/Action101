import os
import numpy as np
from nilearn import image
from canonical_correlation_funcs import extract_roi, run_cca_all_subjects
from stats_funcs import get_pvals_sub, get_pvals_group, save_nifti
from sklearn.manifold import MDS
from scipy.stats import false_discovery_control as fdr
from scipy.stats import ttest_ind
import itertools
from utils.eval_kmeans import evalKMeans
from matplotlib import pyplot as plt

def save_nifti(results, briks, atlas_file='Schaefer200', savepath=''):
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    x,y,z = atlas.get_fdata().shape
    
    image_final = np.squeeze(np.zeros((x,y,z,briks)))

    for r, roi in enumerate(atlas_rois):
        x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
        
        image_final[x_inds, y_inds, z_inds] = results[:, r]

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename(savepath)

    return

if __name__ == '__main__': 

    # Set parameters
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    n_subs = len(sub_list)
    atlas_file = 'Schaefer200' # 'Schaefer100' # 'Schaefer200'
    suffix = '_pcanoz_fullmodel'  

    # Initialize matrix
    pca_subs = np.full((len(sub_list), 200), np.nan)

    # Load pca dict for each subject 
    for s, sub in enumerate(sub_list):
        pca_sub = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/CCA_res_sub-{}_{}.npz'.format(sub, suffix, sub, atlas_file), allow_pickle=True)['pca_dict'].item()

        # Get % explained variance for max n_comps (38)
        for r, roi in enumerate(pca_sub):
            pca_subs[s,r] = pca_sub[roi][-1]

    # Get mean across subs
    pca_subs = np.concatenate((pca_subs, np.expand_dims(np.mean(pca_subs, axis=0), 0)), axis=0)
            
    # Create nifti
    save_nifti(pca_subs, pca_subs.shape[0], atlas_file, '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/CCA_expvar_pca.nii.gz')
    
    print('')
    