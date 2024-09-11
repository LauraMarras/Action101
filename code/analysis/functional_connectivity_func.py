import os
import numpy as np
from nilearn import image
from canonical_correlation_funcs import extract_roi
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import seaborn as sns

def functional_conn(sub_list, condition, atlas_file, rois_list):
    
    # Init correlation matrix
    corr_mats = np.full((len(sub_list), len(rois_list), len(rois_list)), np.nan)
    
    # Iterate over subjects
    for s, sub in enumerate(sub_list):
        
        sub_str = str(sub) if len(str(sub))>=2 else '0'+str(sub)

        # Load data
        data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-{}/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub_str, condition)).get_fdata()
        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/sub-{}_{}_atlas_2orig.nii.gz'.format(sub, atlas_file)).get_fdata()
        
        # Extract rois
        data_rois, n_rois, n_voxels = extract_roi(data, atlas)
        tpoints = data_rois[1].shape[0]

        # Init res mat
        ROI_sub = np.full((len(rois_list), tpoints), np.nan)

        # Iterate over ROIs within mask
        for r, roi in enumerate(rois_list):
            
            roi_avg = np.mean(data_rois[roi], axis=1)
            ROI_sub[r,:] = roi_avg
        
        # Get correlation matrix
        distances = squareform(pdist(ROI_sub, metric='correlation'))
        corr_mats[s] = 1- distances

    # Average across subs
    corr_mat = np.mean(corr_mats, axis=0)
    
    return corr_mat

if __name__ == '__main__':
    
    # Set options and parameters
    condition = 'AV'
    plot = False
    save = False

    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    atlas_file = 'Schaefer200'
    rois_sign = np.loadtxt('{}cca_results/{}/group/fullmodel/significantROIs_{}.txt'.format(global_path, condition, condition)).astype(int)
    
    # Get functional connectivity matrix
    func_conn_mat = functional_conn(sub_list, condition, atlas_file, rois_sign)
    
    # Save functional connectivity matrix
    if save:
        path = '{}/tSNE_clustering/func_conn/'.format(global_path)
        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt('{}func_conn_matrix_{}signROIs.txt'.format(path, condition), func_conn_mat)

    # Plot functional connectivity matrix
    if plot:
        
        # Load ROI labels
        labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[rois_sign-1]
        labels_short = np.array([l[3:] for l in labels])
        
        # Create Figure
        plt.figure(figsize=(15,12))
        sns.heatmap(func_conn_mat, cmap='rocket_r', yticklabels=labels_short, xticklabels=labels_short, square=True)

        plt.title('Functional connectivity matrix based on {} fMRI activity'.format(condition))

        # Save
        path = '{}/tSNE_clustering/func_conn/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        plt.savefig('{}func_conn_matrix_{}signROIs.png'.format(path, condition))