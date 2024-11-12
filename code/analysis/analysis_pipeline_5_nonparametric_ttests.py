import os
import numpy as np
from scipy.stats import false_discovery_control as fdr
from scipy.stats import wilcoxon, friedmanchisquare
import itertools
import pandas as pd
from nilearn import image
from scipy.stats import chi2

def fisher_sum(pvals):
    
    """
    Compute Fisher sum of pvalues across conditions
    
    Inputs:
    - pvals : array, 4d matrix of shape = n_conds by n_rois by n_doms by n_doms, containing pvals of each domain pairing, ROI and condition
    
    Outputs:
    - pval : array, 3d matrix of shape = n_rois by n_doms by n_doms, containing aggregated pvalues
    - t : array, 3d matrix of shape = n_rois by n_doms by n_doms, containing aggregated t statistic
    """
    
    # Define number of tests, in this case = n conditions
    n_test = pvals.shape[0]

    # Calculate t and pval with Fisher's formula
    t = -2 * (np.sum(np.log(pvals), axis=0))
    pvals_summed = 1-(chi2.cdf(t, 2*n_test))

    return pvals_summed, t

def create_nifti(results, ROIsmask, briks, atlas_file, savepath):
    
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

    x,y,z = atlas.get_fdata().shape
    
    image_final = np.squeeze(np.zeros((x,y,z,briks)))

    for r, roi in enumerate(ROIsmask):
        x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
        
        image_final[x_inds, y_inds, z_inds] = results[r]

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename(savepath)

    return

def wilcox_test(ranks):

    # Get ranks in rignt dimension
    if len(ranks.shape) == 2:
        ranks = np.expand_dims(ranks, 1)

    # Get number of ROIs, domains and create domains pairings
    _, n_rois, n_doms = ranks.shape
    combs = list(itertools.combinations(range(n_doms), 2))

    # Initialize res
    wilcox_pvals_mat = np.ones((n_rois, n_doms, n_doms))

    # Iterate over ROIs
    for r in range(n_rois):
    
        # Iterate over domains pairings
        for comb in combs:
            dom1 = ranks[:, r, comb[0]]
            dom2 = ranks[:, r, comb[1]]

            # Run Wilcoxon test
            _, pval = wilcoxon(dom1, dom2, alternative='greater')

            # Assign pvalue to contrast in matrix and 1-pvalue in opposite contrast
            wilcox_pvals_mat[r, comb[0], comb[1]] = pval
            wilcox_pvals_mat[r, comb[1], comb[0]] = 1-pval

    return wilcox_pvals_mat

def maxwin(win_mat, pvals):

    n_rois = win_mat.shape[0]

    # Initialize res
    flags = np.full(n_rois, np.nan)
    win_val = np.full(n_rois, np.nan)
    max_idxs = []

    for r in range(n_rois):
        roi_doms = win_mat[r]
        max_dom = np.max(roi_doms)

        if max_dom:
            idxs = np.where(roi_doms == max_dom)[0]

            if len(idxs) > 1:
                
                minpval = np.argmin(pvals[r])
                max_idxs.append(list(idxs))
            
            else:
                max_idxs.append(idxs[0])
        else:
            max_idxs.append(np.nan)

        # Prendi il dominio che vince di più
        flags[r] = np.argmax(win_mat[r])+1 # Solve issue of ties (argmax takes first index in case of tie results)

        # Prendi il numero di confronti vinti da quel dominio
        win_val[r] = np.max(win_mat[r])

    flags[np.where(win_val==0)] = 0

    return max_idxs, flags, win_val

def run_sim(n_subs, n_rois, n_doms, n_perms, alpha=0.05, seed=0):
    
    # Simulate random results
    np.random.seed(seed=seed)
    rand_res = np.random.rand(n_subs, n_rois, n_doms, n_perms)
    rand_res_ranks = np.argsort(np.argsort(rand_res, axis=1), axis=1)+1
    rand_roi = np.random.randint(0,127)

    # Run wilcox for simulated data
    win_val_rand = np.zeros((n_perms))
    
    for p in range(n_perms):
        _, win_val_rand[p], _ = wilcox_test(rand_res_ranks[:,rand_roi,:,p])
    
    # Get threshold from null distro
    tresh = np.percentile(win_val_rand, q=100-(alpha*100))

    return tresh
    
def wilcox_test_old(ranks, fdr_corr=False):

    # Get number of ROIs, domains and create domains pairings
    _, n_rois, n_doms = ranks.shape
    combs = list(itertools.combinations(range(n_doms), 2))

    # Initialize results matrices
    wilcox_pvals = np.full((n_rois, len(combs)), np.nan)
    wilcox_doms = np.full((n_rois, len(combs)), np.nan)
    wilcox_doms2 = np.full((n_rois, len(combs)), np.nan)
    win_mat = np.zeros((n_rois, n_doms, n_doms))
    flags = np.zeros(n_rois)
    
    # Iterate over domains pairings
    for c, comb in enumerate(combs):
        dom1 = ranks[:, :, comb[0]]
        dom2 = ranks[:, :, comb[1]]

        _, wilcox_pvals[:, c] = wilcoxon(dom1, dom2, axis=0)
        
        # Measure direction
        # previous way
        diff = dom1 - dom2
        wilcox_doms[:, c] = (np.median(diff, axis=0)<0)*1

        # New method
        abs_differences = np.abs(diff)
        signs = np.sign(diff)
        ranks_diffs = np.argsort(np.argsort(abs_differences, axis=0), axis=0) + 1  
        
        T_plus = np.full(n_rois, np.nan)
        T_minus = np.full(n_rois, np.nan)
        for r in range(n_rois):
            T_plus[r] = np.sum(ranks_diffs[:,r][signs[:,r] > 0])
            T_minus[r] = np.sum(ranks_diffs[:,r][signs[:,r] < 0])

        wilcox_doms2[:, c] = 1 - (T_plus>T_minus)
        if np.any(T_plus == T_minus):
            wilcox_doms2[np.where(T_minus == T_plus)[0],c] = np.nan

    # Get significant ROIs
    if fdr_corr:
        wilcox_pvals = fdr(wilcox_pvals)
    
    sign_wilc_idxs = np.unique(np.where(wilcox_pvals<0.05)[0])

    # Get win matrix
    for r in sign_wilc_idxs:
        for c, comb in enumerate(combs):

            if wilcox_doms[r,c] == 0:
                win_mat[r, comb[0], comb[1]] += 1
            else:
                win_mat[r, comb[1], comb[0]] += 1
        
    # Get maxwin for each ROI
    maxwin_mat = np.sum(win_mat, axis=-1)
    flags[sign_wilc_idxs] = np.argmax(maxwin_mat[sign_wilc_idxs,:], axis=-1)+1 # Solve issue of ties (argmax takes first index in case of tie results)

    maxwin = np.max(win_mat, axis=-1)

    return flags, maxwin_mat, wilcox_pvals, combs, maxwin

if __name__ == '__main__': 
    
    # Set parameters
    conditions = ['AV', 'vid', 'aud']
    n_perms = 1000
    simulate = False
   
    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/' # 'C:/Users/SemperMoMiLab/Documents/Repositories/Action101/data/' #
    atlas_file = 'Schaefer200'
    roislabels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)
    rois_sign_AV = np.load('{}cca_results/AV/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path), allow_pickle=True)['rois_list']
    n_rois = rois_sign_AV.shape[0]

    # Load task models
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    wilcox_pvals = np.full((len(conditions), n_rois, n_doms, n_doms), np.nan)

    # Iterate over conditions
    for c, condition in enumerate(conditions):
        
        sub_list = sub_lists[condition]
        n_subs = len(sub_list)

        # Load group results
        results_singledoms = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['results_subs_sd']
        
        # Get ranks
        rois_ranks = np.argsort(np.argsort(results_singledoms, axis=1), axis=1)+1 # For each subject and for each domain, get ranks of R2 across ROIs
    
        # Run Wilcoxon
        wilcox_pvals[c] = wilcox_test(rois_ranks)
        
    # Sum pvals with fisher
    pvals_summed, t = fisher_sum(wilcox_pvals)

    # Threshold
    alpha = 0.05    
    pvals_tresh = 1*(pvals_summed < alpha)

    # Maxwin
    win_mat = np.sum(pvals_tresh, axis=-1)
    pvals_win = np.sum(pvals_summed, axis=-1)

    


    # Get treshold for multiple comparisons
    if simulate:
        tresh = run_sim(n_subs, n_rois, n_doms, n_perms)
    else:
        tresh = 2
    
    # Assign flag to winning domain for ROIs above threshold
    sign_idxs = np.where(win_val > tresh)[0]
    flags_sign = np.zeros(n_rois)
    flags_sign[sign_idxs] = flags[sign_idxs]

    # Save
    path = '{}ttest_nonparam/'.format(global_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # Create dataframe
    doms_lab = domains_list
    doms_lab.insert(0, 'None')
    flags_labels = np.array([doms_lab[int(f)] for f in flags_sign])

    results = pd.DataFrame((rois_sign_AV[sign_idxs], flags_labels[sign_idxs], win_val[sign_idxs].astype(int))).T
    results['roilabel'] = results[0].apply(lambda x: roislabels[x-1])
    results.rename({0:'roi', 1:'domain', 2:'wins'}, axis=1, inplace=True)
    
    results[['roi', 'roilabel', 'domain', 'wins']].to_csv('{}wilcoxon_{}.csv'.format(path, condition), header=['roi', 'roilabel', 'domain', 'wins'], index=False)

    # Create Nifti
    create_nifti(np.vstack((flags, win_val)).T, rois_sign_AV, 2, atlas_file, '{}wilcoxon_{}.nii'.format(path, condition))