import os
import numpy as np
from nilearn import image
from canonical_correlation_funcs import extract_roi, run_cca_all_subjects
from stats_funcs import get_pvals_sub, get_pvals_group, save_nifti
from sklearn.manifold import MDS, TSNE
from scipy.stats import false_discovery_control as fdr
from scipy.stats import ttest_ind, wilcoxon, ttest_rel
import itertools
from utils.eval_kmeans import evalKMeans
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


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

def ttest_rel_domains(results, dom_combs, dom_combs_str, fdr_opt=True, save=False):
    
    # Initialize results matrices
    ts = np.empty((results.shape[1], len(dom_combs)))
    pvals = np.empty((results.shape[1], len(dom_combs)))

    # Run ttest_rel for each ROI and each domains pairing
    for roi in range(results.shape[1]):  
        for comb in dom_combs:
            d1 = results[:, roi, comb[0]]
            d2 = results[:, roi, comb[1]]
            t, pval = ttest_rel(d1, d2)
            ts[roi,comb] = t 
            pvals[roi,comb] = pval
    
    # FDR correction for multiple comparisons
    if fdr_opt:
        pvals = np.reshape(fdr(np.ravel(pvals)), pvals.shape)

    # Save results
    if save:
        np.savez('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/ttest/ttest_rel_cca', tstat=ts, pvals=pvals, domains_contrasts=dom_combs_str, FDRcorrection=fdr_opt)

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

def create_ttest_nifti(ts, pvals, dom_combs, dom_combs_str):
        for c, comb in enumerate(dom_combs):
            cres = np.expand_dims(np.vstack((ts[:,c], pvals[:,c])).T, axis=0)
            path = '/data1/Action_teresi/CCA/cca_results/group/ttest/ttest_{}.nii.gz'.format(dom_combs_str[c][0][:5] + 'vs' + dom_combs_str[c][1][:5])
            save_nifti(cres, 2, atlas_file, path, ROIs_2keep+1)

        return

def functional_conn(sub_list, atlas_file, rois_list):

    # Init correlation matrix
    corr_mats = np.full((len(sub_list), len(rois_list), len(rois_list)), np.nan)
    
    # Iterate over subjects
    for s, sub in enumerate(sub_list):
        
        # Load data
        data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-AV/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub)).get_fdata()
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
        corr_mats[s] = distances

    # Average across subs
    corr_mat = np.mean(corr_mats, axis=0)
    
    return corr_mat

if __name__ == '__main__': 
    
    # Set parameters
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    atlas_file = 'Schaefer200'
    plot_clust = False
    suffix = '_pca_variancepart'
    plot_tsne = False
    metric = 'euclidean'
    plot_funcconn = False

    # Load group results and filter ROIs with significant R2 for full model
    pvals_group_fm = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group_pcanoz_fullmodel_6doms_{}_maxT.npz'.format(atlas_file), allow_pickle=True)['pvals_group']
    ROIs_2keep = np.unique(np.where(np.squeeze(pvals_group_fm) < 0.05)[0])
    results_group = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_group{}_{}.npz'.format(suffix, atlas_file), allow_pickle=True)['results_group'][ROIs_2keep,:]

    results_filtered_vp = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/CCA_res_allsubs_singledoms_filtered.npy', allow_pickle=True)
    results_vp_avg = results_filtered_vp[-1,:,:]
    results_vp = results_filtered_vp[:-1,:,:]

    # Get functional connectivity matrix for significant ROIs
    # func_conn_mat = functional_conn(sub_list, atlas_file, ROIs_2keep+1)
    
    # # Save functional connectivity matrix
    # np.save('/home/laura.marras/Documents/Repositories/Action101/data/func_connectivity_matrix_CCArois', func_conn_mat)
    # np.savetxt('/data1/Action_teresi/CCA/func_connectivity_matrix_CCArois.txt', func_conn_mat)

    # Load functional connectivity matrix
    func_conn_mat = np.load('/home/laura.marras/Documents/Repositories/Action101/data/func_connectivity_matrix_CCArois.npy')

    # Plot functional connectivity matrix
    if plot_funcconn:
        plt.figure(figsize=(15,12))
        labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[ROIs_2keep]
        labels_short = np.array([l[3:] for l in labels])
        sns.heatmap(func_conn_mat, yticklabels=labels_short, xticklabels=labels_short, square=True)
        plt.savefig('func_conn.png')

    # tSNE
    tsne = TSNE(random_state=0, metric='precomputed', init='random', perplexity=5, n_iter=5000)
    results_tsne = tsne.fit_transform(func_conn_mat)
    
    # tSNE
    # tsne = TSNE(random_state=0, metric=metric, init='random', perplexity=5, n_iter=5000)
    # results_tsne = tsne.fit_transform(results_vp_avg)

    if plot_tsne:
        plt.figure()
        scatt = plt.scatter(results_tsne[:,0], results_tsne[:,1])
        plt.title('tSNE on {} distance btw  ROIs R2'.format(metric))
        plt.suptitle('KL-divergence = {}'.format(tsne.kl_divergence_))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/clustering/tsne_{}_new.png'.format(metric))
    
    # Clustering
    # n_clust_max=5
    # silhouette_avg, clusters_labels = clustering(squareform(pdist(results_tsne, metric=metric)), n_clust_max, atlas_file)
    # clust = np.argmax(silhouette_avg)

    # Clustering
    # n_clust_max=15
    # silhouette_avg, clusters_labels = clustering(results_tsne, n_clust_max, atlas_file)
    # clust = np.argmax(silhouette_avg)
    
    # Plot
    if plot_clust:
        # Load ROI labels
        labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[ROIs_2keep]
        hemispheres = np.array([0 if l[0]=='R' else 1 for l in labels])
        labels_short = np.array([l[3:] for l in labels])
        right = np.where(hemispheres==0)[0]
        left = np.where(hemispheres)[0]

        plt.figure(figsize=(9.6, 6.4))
        scatt = plt.scatter(results_tsne[right,0], results_tsne[right,1], c=clusters_labels[clust, right], marker='o')
        scattL = plt.scatter(results_tsne[left,0], results_tsne[left,1], c=clusters_labels[clust, left], marker='v')
        plt.legend(*scattL.legend_elements(), loc='lower right', title='Clusters', ncol=2, frameon=False, borderpad=-1, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.1, handletextpad=-0.5)
        plt.title('Clustering ROIs on tSNE space from {}'.format(metric))
        plt.suptitle('Silhouette score for {} clusters = {}'.format(clust+2, silhouette_avg[clust]))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for i, txt in enumerate(labels_short):
            ax.annotate(txt, (results_tsne[i,0]+0.5, results_tsne[i,1]+0.5), size='xx-small')

        plt.savefig('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/clustering/tsne_{}_kmeans_{}clusters.png'.format(metric, clust+2))

    # Save clustering res to nifti
    # clust_to_nifti = np.expand_dims(clusters_labels[clust], axis=0)
    # save_nifti(clust_to_nifti, 1, atlas_file, '/data1/Action_teresi/CCA/cca_results/group/CCA_tsne_euclidean_kmeans.nii.gz', ROIsmask=ROIs_2keep+1)

    # Load task models
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())
    n_predictors = np.sum([domains[d].shape[1] for d in domains_list])
    dom_combs = list(itertools.combinations(range(n_doms), 2))
    dom_combs_str = list(itertools.combinations(domains_list, 2))
    
    # Run t-tests
    #ts, pvals = ttest_rel_domains(results_vp, dom_combs, dom_combs_str, fdr_opt=True, save=True)
    
    # Save nifti
    # create_ttest_nifti(ts, 1-pvals, dom_combs, dom_combs_str)

    ts = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/ttest/ttest_rel_cca.npz', allow_pickle=True)['tstat']
    pvals = np.load('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/ttest/ttest_rel_cca.npz', allow_pickle=True)['pvals']

    doms_color = {'space': '#ff595e', 'movement':'#ff924c', 'agent_objective':'#ffca3a', 'social_connectivity':'#8ac926', 'emotion_expression':'#1982c4', 'linguistic_predictiveness':'#6a4c93'}
    
    for c, comb in enumerate(dom_combs_str):
        
        cmap = LinearSegmentedColormap.from_list('mycmap', [doms_color[comb[0]], doms_color[comb[1]], '#adb5bd'], N=3)
    
        color_coding = np.array([0 if t>0  else 1 for t in ts[:,c]])
        color_coding[np.where(pvals[:,c]>0.05)] = '2'
        
        point_labels = np.array(['{}'.format(comb[0]) if t >0 else '{}'.format(comb[1]) for t in ts[:,c]])
        point_labels[np.where(pvals[:,c]>0.05)] = 'ns'

        # Load ROI labels
        labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[ROIs_2keep]
        hemispheres = np.array([0 if l[0]=='R' else 1 for l in labels])
        labels_short = np.array([l[3:] for l in labels])
        right = np.where(hemispheres==0)[0]
        left = np.where(hemispheres)[0]

        plt.figure(figsize=(9.6, 6.4))
        scatt = plt.scatter(results_tsne[right,0], results_tsne[right,1], c=color_coding[right], cmap=cmap, marker='o')
        scattL = plt.scatter(results_tsne[left,0], results_tsne[left,1], c=color_coding[left], cmap=cmap, marker='v')
        
        # Create legend

        legend_elements = [Line2D([0], [0], color='w', markerfacecolor=doms_color[comb[0]], marker='o', label=comb[0]),
                            Line2D([0], [0], color='w', markerfacecolor=doms_color[comb[1]], marker='o', label=comb[1]),
                            Line2D([0], [0], color='w', markerfacecolor='#adb5bd', marker='o', label='ns'),
                            Line2D([0], [0], color='w', markerfacecolor='k', marker='o', label='right'),
                            Line2D([0], [0], color='w', markerfacecolor='k', marker='v', label='left')]
                   
        plt.legend(handles=legend_elements, loc='upper left', ncol=1, frameon=False, borderpad=-1, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.1, handletextpad=-0.5)
                
        
        plt.title('T-Test on tSNE space from {}\n{}'.format('functional connectivity', comb[0]+' vs '+comb[1]))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for i, txt in enumerate(labels_short):
            ax.annotate(txt, (results_tsne[i,0]+0.5, results_tsne[i,1]+0.5), size='xx-small')

        plt.savefig('/home/laura.marras/Documents/Repositories/Action101/data/cca_results/clustering/ttest_{}_vs_{}.png'.format(comb[0], comb[1]))


        print('')
