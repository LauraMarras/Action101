import os
import numpy as np
import itertools
from scipy.stats import false_discovery_control, wilcoxon, ranksums, chi2
from nilearn import image

def fisher_sum(pvals, axis=0):
    
    """
    Compute Fisher sum of p values across the specified axis
    
    Inputs:
    - pvals : array, a 2D or higher matrix containing p values
    - axis : int, specify the axis along to which compute the aggregated p values; default = 0
    
    Outputs:
    - pvals_summed : array, 1D or higher matrix containing aggregated p values
    - t : array, 1D or higher matrix containing aggregated t statistic
    """
    
    # Define number of tests, in this case = n conditions
    n_test = pvals.shape[axis]

    # Calculate t and pval with Fisher's formula
    t = -2 * (np.sum(np.log(pvals), axis=axis))
    pvals_summed = 1-(chi2.cdf(t, 2*n_test))

    return pvals_summed, t

def wilcox_test(data, paired=True):

    """
    Compute pairwise Wilcoxon signed-rank tests between domains for each ROI
    
    Inputs:
    - data : array, 2D or 3D matrix with shape (n_subjects, n_rois, n_doms), if 2D it is considered a single ROI and it will be expanded to 3D along a new axis
    
    Outputs:
    - pvals : array, 3D matrix of shape (n_rois, n_doms, n_doms) containing p-values from the Wilcoxon tests (1-tailed)
    """

    # Get data in rignt dimension
    if data.ndim == 2:
        data = np.expand_dims(data, 1)

    # Get number of ROIs, domains and create domains pairwise contrasts
    _, n_rois, n_doms = data.shape
    pairwise_contrasts = list(itertools.combinations(range(n_doms), 2))
    n_tests = 2*(len(pairwise_contrasts))

    # Initialize res
    pvals = np.ones((n_rois, n_doms, n_doms))

    # Iterate over ROIs
    for r in range(n_rois):
    
        # Iterate over domains pairwise contrasts
        for domA, domB in pairwise_contrasts:
            domA_data = data[:, r, domA]
            domB_data = data[:, r, domB]

            # Run Wilcoxon test
            if paired:
                _, pvals[r, domA, domB] = wilcoxon(domA_data, domB_data, alternative='greater')
                _, pvals[r, domB, domA] = wilcoxon(domB_data, domA_data, alternative='greater')
            else:
                _, pvals[r, domA, domB] = ranksums(domA_data, domB_data, alternative='greater')
                _, pvals[r, domB, domA] = ranksums(domB_data, domA_data, alternative='greater')
    return pvals, n_tests

def maxwin(contrasts, pvals=None):

    """
    Get max winning domain(s) for each ROI
    
    Inputs:
    - contrasts : array, 2D matrix of shape (n_rois, n_domains), containing number of contrasts won for each ROI and domain
    - pvals : array, 2D matrix of shape (n_rois, n_domains), containing summed significant p-values for each ROI and domain, to use for selecting single domain in case more than 1 domains have equal number of max won contrasts; default = None
    
    Outputs:
    - flags : list, list of len (n_rois) arrays, containing arrays with domain(s) flag for each ROI
    - n_woncontrasts : array, 1D array containing max number of won contrasts for each ROI
    - n_flags : array, 1D array containing 
    """

    # Get number of ROIs
    n_rois = contrasts.shape[0]

    # Initialize res
    flags = []
    n_woncontrasts = np.zeros(n_rois)
    n_flags = np.zeros(n_rois)

    # Iterate over ROIs
    for r, roi in enumerate(contrasts):

        # Check if there is at least one significant contrast for that ROI
        if np.any(roi):
            
            # Get domain(s) with highest number of won contrasts
            idxs = np.where(roi == np.max(roi))[0]

            # Store n of winning domain(s), number of won contrasts and domain(s) flags
            n_flags[r] = len(idxs)
            n_woncontrasts[r] = np.max(roi)
            domain = idxs + 1

            # Select winning domain based on lower summed pvalues of all significant contrasts
            if pvals is not None:
                minpval_idx = np.argmin(pvals[r, idxs])
                domain = idxs[minpval_idx] + 1

        else:
            domain = np.nan

        flags.append(domain)

    return flags, n_woncontrasts.astype(int), n_flags.astype(int)

def simulate_pvals(shape, seed=0, path=None):
    
    """
    Simulate random ranks, perform Wilcoxon signed-rank tests, and aggregate p-values using Fisher's method
    
    Inputs:
    - shape : tuple, shape of the simulation data (n_subs, n_conds, n_rois, n_doms, n_perms)
    - seed : int, random seed; default = 0
    - path : str, optional file path to save the resulting p-values; default = None (i.e. don't save)
    
    Outputs:
    - pvals : array, 4D matrix of shape (n_perms, n_conds, n_doms, n_doms) containing pairwise p-values from Wilcoxon signed-rank tests for each permutation and condition
    - fisher_pvals : array, 3D matrix of aggregated p-values across conditions, computed using Fisher's method
    """

    # Simulate random results
    np.random.seed(seed=seed)
    n_subs, n_conds, n_rois, n_doms, n_perms = shape
    rand_res = np.random.rand(n_conds, n_subs, n_rois, n_doms, n_perms)
    rand_res_data = np.argsort(np.argsort(rand_res, axis=2), axis=2)+1

    # Simulate test on single ROI 
    rand_roi = np.random.randint(0, n_rois)
    
    # Initialize pvals matrix
    pvals = np.full((n_perms, n_conds, n_doms, n_doms), np.nan)

    # Iterate over conditions
    for c in range(n_conds):
        for p in range(n_perms):
            # Run Wilcoxon
            pvals[p, c], _ = wilcox_test(rand_res_data[c, :, rand_roi, :, p])
        
    # Sum pvals with fisher
    fisher_pvals, _ = fisher_sum(pvals, axis=1)

    # Save pvals
    if path:
        np.savez('{}wilcox_simulation{}_pvals'.format(path, n_perms), pvals=pvals, fisher_pvals=fisher_pvals)

    return pvals, fisher_pvals

def threshold(pvals, n_bootstraps=127, alpha=0.05, q=95, seed=0):
    
    """
    Compute a statistical threshold for p-values using bootstrapping and MaxT correction
    
    Inputs:
    - pvals : array, 1D array of p-values (null distribution)
    - n_bootstraps : int, number of bootstrap samples to generate for constructing the null distribution; default = 127 (n_rois)
    - alpha : float, significance level used for thresholding the p-values; default = 0.05
    - q : int, percentile value to extract the threshold from the null distribution; default = 95

    Outputs:
    - tresh : float, the computed threshold derived from the null distribution of maximum statistics
    """

    # Threshold
    pvals_tresholded = 1*(pvals < alpha)

    # Maxwin
    contrasts = np.sum(pvals_tresholded, axis=-1)
    _, n_woncontrasts, _ = maxwin(contrasts)

    # Bootstrapping
    np.random.seed(seed=seed)
    bootstrapped_contrasts = np.random.choice(n_woncontrasts, size=(n_bootstraps), replace=True)

    # MaxT
    max_bootstrap = np.max(bootstrapped_contrasts, axis=0)

    # Get threshold from null distro
    tresh = np.percentile(max_bootstrap, q=q)

    return tresh

def create_nifti(results, ROIsmask, atlas_file, savepath):
    
    """
    Generate and save a NIfTI image from a set of results mapped to specified ROIs
    
    Inputs:
    - results : array, 1D or 2D array of shape (n_rois, n_bricks) containing value to be assigned to all voxels within each ROI
    - ROIsmask : array, 1D array of shape (n_rois), containing ROI identifiers (numbers) in atlas
    - atlas_file : str, file path of the atlas
    - savepath : str, file path where to save the output NIfTI image
    """

    # Load atlas and get volume dimensions
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

    x,y,z = atlas.get_fdata().shape
    
    # Get number of briks
    if results.ndim == 1:
        results = np.expand_dims(results, axis=-1)
    briks = results.shape[1]

    # Initialize volume
    image_final = np.squeeze(np.zeros((x,y,z,briks)))

    # Iterate over ROIs and assign value from results
    for r, roi in enumerate(ROIsmask):
        x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
        
        image_final[x_inds, y_inds, z_inds] = results[r]
    
    # Create nifti and save
    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=True)
    img.to_filename(savepath)

    return

if __name__ == '__main__':
    
    # Set parameters
    conditions = ['AV', 'vid', 'aud']
    n_perms = 10000
    simulate = False
    create_nifti_ranks = True
    paired = True 
    test = 'wilcox' if paired else 'ranksum'
   
    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/' # 'C:/Users/SemperMoMiLab/Documents/Repositories/Action101/data/' #
    atlas_file = 'Schaefer200'
    roislabels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)
    rois_sign_AV = np.load('{}cca_results/AV/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path), allow_pickle=True)['rois_list']
    n_rois = rois_sign_AV.shape[0]

    # Save path
    path = '{}ttest_nonparam/'.format(global_path) #'/data1/Action_teresi/CCA/contrasts/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Create nifti of group ranks for each domain and condition
    if create_nifti_ranks:
        
        for c, condition in enumerate(conditions):
            
            # Get ranks of group results
            results_singledoms = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['results_subs_sd']
            
            results_group = np.mean(results_singledoms, axis=0)
            ranks_avg = np.argsort(np.argsort(results_group, axis=0), axis=0)+1

            # Save nifti
            create_nifti(ranks_avg, rois_sign_AV, atlas_file, '{}avg_ranks_{}.nii'.format(path, condition))

    # Load task models
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    # Estimate threshold for FPR through simulation
    if simulate:
        # Load simulated fisher pvalues if already calculated
        try:
            fisher_pvals_sim = np.load('{}wilcox_simulation{}_pvals.npz'.format(path, n_perms))['fisher_pvals']
        # Else simulate them
        except FileNotFoundError:
            pvals_sim, fisher_pvals_sim = simulate_pvals(shape=(10, 3, n_rois, n_doms, n_perms), path=path)
    
        tresh = threshold(fisher_pvals_sim, alpha=0.005)

    # Load Wilcoxon fisher pvalues if already calculated
    try:
        fisher_qvals = np.load('{}{}_pvals.npz'.format(path, test))['fisher_qvals']
    
    # Else Calculate them
    except FileNotFoundError:
        # Initialize pvals matrix
        wilcox_pvals = np.full((len(conditions), n_rois, n_doms, n_doms), np.nan)

        # Iterate over conditions
        for c, condition in enumerate(conditions):
            
            sub_list = sub_lists[condition]
            n_subs = len(sub_list)

            # Load group results
            results_singledoms = np.load('{}cca_results/{}/group/single_doms/CCA_R2_allsubs_singledoms.npz'.format(global_path, condition), allow_pickle=True)['results_subs_sd']

            # Get ranks
            rois_ranks = np.argsort(np.argsort(results_singledoms, axis=1), axis=1)+1 # For each subject and for each domain, get data of R2 across ROIs

            # Run Wilcoxon
            wilcox_pvals[c], n_tests = wilcox_test(rois_ranks, paired)
            
        # Sum pvals with fisher
        fisher_pvals, t = fisher_sum(wilcox_pvals)

        # Correct for multiple comparisons
        fisher_qvals = false_discovery_control(fisher_pvals)

        # Save fisher pvals and qvals
        np.savez('{}{}_pvals'.format(path, test), fisher_pvals=fisher_pvals, fisher_qvals=fisher_qvals)

    # Threshold
    alpha = 0.05
    pvals_tresholded = 1*(fisher_qvals < alpha)

    # Maxwin
    contrasts = np.sum(pvals_tresholded, axis=-1)
    flags, n_woncontrasts, n_flags = maxwin(contrasts)

    # Create nifti for each domain
    domains_ncontrasts = np.full((n_rois, n_doms), np.nan)
    for domain in range(n_doms):
        domains_ncontrasts[:, domain] = np.array([n_woncontrasts[r] if np.isin(domain+1, doms) else 0 for r, doms in enumerate(flags)])

    create_nifti(domains_ncontrasts, rois_sign_AV, atlas_file, '{}{}_q{}.nii'.format(path, test, str(alpha)))