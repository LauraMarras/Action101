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

def evaluate_nvoxels_rois(sub_list, atlas_file, n_predictors, print_opt=True, save=False):
    
    rois_to_exclude = []
    n_voxs_forex = []

    for sub in sub_list:
        # Load Atlas
        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/sub-{}_{}_atlas_2orig.nii.gz'.format(sub, atlas_file)).get_fdata()

        # Load data
        data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-AV/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub)).get_fdata()

        # Get ROI info
        _, _, n_voxels_rois = extract_roi(data, atlas)

        # Get number of ROIs with less voxels than the full_model predictors
        n_ROIs_below = np.sum(list(n_voxels_rois.values()) < n_predictors)
        ROIs_below = [k for k,v in n_voxels_rois.items() if v < n_predictors]
        n_vox_ROIs_below = [n_predictors - v for k,v in n_voxels_rois.items() if v < n_predictors]

        if print_opt:
            print('sub-{}'.format(sub))
            print('     n ROIs with less than {} voxels: {}'.format(n_predictors, n_ROIs_below))
            print('     ROIs id: {}'.format(ROIs_below))
            print('     missing voxels: {}'.format(n_vox_ROIs_below))

        for r, roi in enumerate(ROIs_below):
            rois_to_exclude.append(roi)
            n_voxs_forex.append(n_vox_ROIs_below[r])

    rois_nvoxs_subs = {roi: np.array(n_voxs_forex)[np.where(np.array(rois_to_exclude) == roi)].tolist() for roi in np.unique(np.array(rois_to_exclude))}
    rois_nsubs = dict((x, rois_to_exclude.count(x)) for x in set(rois_to_exclude))
    
    # Create nifti 
    if save:
        # Load Atlas
        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
        x,y,z = atlas.get_fdata().shape

        # assign number of subject where ROI is too small as value
        image_final = np.zeros((x, y, z))

        for roi, val in rois_nsubs.items():
            x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
            image_final[x_inds, y_inds, z_inds] = val
        
        # Create nifti
        img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
        img.to_filename('/data1/Action_teresi/CCA/atlas/{}_rois_toexclude.nii.gz'.format(atlas_file))

    return rois_nsubs, rois_nvoxs_subs

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

def clustering(results_group, n_clust_max, atlas_file):

    silhouette_avg, clusters_labels = evalKMeans(range(2, n_clust_max+1), results_group, print_otp=False)
    clusters_labels += 1

    # Load Atlas
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    x,y,z = atlas.get_fdata().shape
        
    # Initialize volume
    image_final = np.zeros((x, y, z, n_clust_max+1-2))

    # Create volume with cluster values for each k
    for k in range(n_clust_max+1-2):
        for r in range(results_group.shape[0]):
            x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==r+1)
            image_final[x_inds, y_inds, z_inds, k] = clusters_labels[k,r]
        
    # Create nifti
    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename('/data1/Action_teresi/CCA/{}_clustering.nii.gz'.format(atlas_file))

    return silhouette_avg, clusters_labels

if __name__ == '__main__': 

    # Set parameters
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    n_subs = len(sub_list)
    
    atlas_file = 'Schaefer200' #'Schaefer100' #
    suffix = '_pca'

    # Clustering
    n_clust_max=10
    results_group = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['results_group']
    silhouette_avg, clusters_labels = clustering(results_group, n_clust_max, atlas_file)

    mds = MDS().fit_transform(results_group)

    # Plot
    for clust in range(clusters_labels.shape[0]):
        plt.figure()
        scatt = plt.scatter(mds[:,0], mds[:,1], c=clusters_labels[clust])
        plt.legend(*scatt.legend_elements(), loc='lower right', title='Clusters', ncol=2, frameon=False, borderpad=-1, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.1, handletextpad=-0.5)
        plt.title('Clustering ROIs')
        plt.suptitle('Silhouette score for {} clusters = {}'.format(clust+2, silhouette_avg[clust]))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/clustering/MDS_kmeans_{}clusters.png'.format(clust+2))

    # Load task models
    # domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    # domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    # n_doms = len(domains.keys())
    # n_predictors = np.sum([domains[d].shape[1] for d in domains_list])
    # dom_combs = list(itertools.combinations(range(n_doms), 2))
    # dom_combs_str = list(itertools.combinations(domains_list, 2))

    # Evaluate atlas ROIs dimension
    #rois_nsubs, rois_nvoxs_subs = evaluate_nvoxels_rois(sub_list, atlas_file, n_predictors, print_opt=True, save=False)

    # Load single subject results
    #results = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['results'][:,:,0,:]
    #results_group = np.mean(results, axis=0)

    # Run t-tests
    #ts, pvals = ttest_domains(results, dom_combs, dom_combs_str, fdr_opt=True, save=True)

    

    
    print('')