import os
import numpy as np
from scipy.stats import false_discovery_control as fdr
from scipy.stats import wilcoxon, friedmanchisquare
import itertools

if __name__ == '__main__': 
    
    # Set parameters
    condition = 'AV'
   
    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/' # 'C:/Users/SemperMoMiLab/Documents/Repositories/Action101/data/' #
    atlas_file = 'Schaefer200'

    # Load group results
    results_singledoms = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['results_subs_sd']
    rois_sign = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['rois_list']

    # Load task models
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())
    n_predictors = np.sum([domains[d].shape[1] for d in domains_list])

    dom_combs = list(itertools.combinations(range(n_doms), 2)) #list(itertools.product(range(n_doms), range(n_doms))) #
    dom_combs_str = np.array(list(itertools.combinations(domains_list, 2))) #np.array(list(itertools.product(domains_list,domains_list))) #
    
    # Run t-tests
    path = '{}ttest_nonparam/'.format(global_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # Get ranks
    rois_ranks = np.argsort(np.argsort(results_singledoms, axis=1), axis=1)+1 # For each subject and for each domain, get ranks of R2 across ROIs

    # Run Friedman test
    friedman_stats, friedman_pvals = friedmanchisquare(rois_ranks[:,:,0], rois_ranks[:,:,1], rois_ranks[:,:,2], rois_ranks[:,:,3], rois_ranks[:,:,4], rois_ranks[:,:,5], axis=0)

    # Correct for multiple comparisons
    friedman_qvals = fdr(friedman_pvals)

    # Run pairwise comparisons with wilcoxon for significant ROIs
    idx_fried_sign = np.where(friedman_qvals<0.05)[0]
    rois_fried_sign = rois_sign[idx_fried_sign]
    
    # Initialize results mat
    wilcox_stats = np.full((len(rois_fried_sign), len(dom_combs)), np.nan)
    wilcox_pvals = np.full((len(rois_fried_sign), len(dom_combs)), np.nan)
    wilcox_doms = np.full((len(rois_fried_sign), len(dom_combs)), np.nan)
    

    # Iterate over doms combinations
    for d, comb in enumerate(dom_combs):
        dom1 = rois_ranks[:, idx_fried_sign, comb[0]]
        dom2 = rois_ranks[:, idx_fried_sign, comb[1]]

        wilcox_stats[:, d], wilcox_pvals[:, d] = wilcoxon(dom1, dom2, axis=0)
        wilcox_doms[:, d] = np.array([comb[i] for i in (np.median(dom1-dom2, axis=0)<0)*1])

    wilcox_doms = wilcox_doms.astype(int)
    
    # Correct for multiple comparisons
    wilcox_qvals = fdr(wilcox_pvals)

    rois_wilcox_sign = rois_fried_sign[np.where(wilcox_qvals<0.05)[0]]
    combs_wilcox_sign = np.array(dom_combs_str)[np.where(wilcox_qvals<0.05)[1]]
    windom_wilcox_sign = combs_wilcox_sign[wilcox_doms[np.where(wilcox_qvals<0.05)]]

    for r, roi in enumerate(rois_wilcox_sign):
        print(roi)
        print(combs_wilcox_sign[r])
        print(windom_wilcox_sign[r])

    print('d')