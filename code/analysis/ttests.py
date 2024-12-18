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

def ttest_domains(results, dom_combs, dom_combs_str, test2use='rel', fdr_opt=True, save=False, path=''):
    
    # Initialize results matrices
    ts = np.empty((results.shape[1], len(dom_combs)))
    pvals = np.empty((results.shape[1], len(dom_combs)))

    # Run ttest_rel for each ROI and each domains pairing
    for roi in range(results.shape[1]):  
        for comb in dom_combs:
            d1 = results[:, roi, comb[0]]
            d2 = results[:, roi, comb[1]]

            if test2use == 'rel':
                t, pval = ttest_rel(d1, d2)
            elif test2use == 'rel':
                t, pval = ttest_ind(d1, d2)

            ts[roi,comb] = t 
            pvals[roi,comb] = pval
    
    # FDR correction for multiple comparisons
    if fdr_opt:
        pvals = np.reshape(fdr(np.ravel(pvals)), pvals.shape)

    # Save results
    if save:
        np.savez('{}ttest_{}{}'.format(path, test2use, '_fdr' if fdr_opt else ''), tstat=ts, pvals=pvals, domains_contrasts=dom_combs_str, FDRcorrection=fdr_opt)

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

if __name__ == '__main__': 
    
    # Set parameters
    condition = 'AV'
    full_model_opt = True # full_model vs variance_partitioning (if False run Variance partitioning)

    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'

    atlas_file = 'Schaefer200'

    # Load group results
    results_singledoms = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['results_subs_sd']
    rois_sign = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['rois_list']

    # Load task models
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())
    n_predictors = np.sum([domains[d].shape[1] for d in domains_list])
    dom_combs = list(itertools.combinations(range(n_doms), 2))
    dom_combs_str = list(itertools.combinations(domains_list, 2))
    
    # Run t-tests
    path = '{}ttest/'.format(global_path)
    if not os.path.exists(path):
                os.makedirs(path)

    ts, pvals = ttest_domains(results_singledoms, dom_combs, dom_combs_str, test2use='rel', fdr_opt=True, save=True, path=path)
    
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
