import numpy as np
from scipy.stats import pareto, chi2
from matplotlib import pyplot as plt
import os
from nilearn import image


def pareto_right(results, critical_p=0.05, tail_percentile=0.9, plot=None):

    """
    Get p value using pareto for a single domain
    
    Inputs:
    - results : array, 1d matrix of shape = n_perms containing R2 values for each permutation
    - critical_p : float, alpha level; default = 0.05
    - tail_percentile : float, percentile of tail to model with Pareto; default = 0.9
    - plot : str, ROI folder and domain name for wich to plot and save pareto; default = None

    Outputs:
    - pvalue : float
    - critical_values_at_p : float
    """

    # Define critical value, null distribution and tail
    null_results = results[1:]
    critical_value = results[0]
    n_permutations = null_results.shape[0]
    q = np.percentile(null_results, tail_percentile*100)
    right_tail = null_results[null_results>q] - q # Recentered
    effect_eps = np.max(right_tail)/(n_permutations/len(right_tail))

    # Fit pareto over right tail
    kHat, loc, sigmaHat = pareto.fit(right_tail, floc=0)

    # Verify that Pareto can be estimated
    if kHat > -0.5 and (critical_value-q) > 0:
        print('Estimating Pareto')
        
        estimated_cum = pareto.cdf(critical_value-q, kHat, loc, sigmaHat)
        pvalue = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

        if pvalue < (np.finfo(np.float64).eps/2):
            bins = np.arange(0, critical_value-q, effect_eps)
            pvalue_emp = np.full((len(bins)), np.nan)
            for b, bin in enumerate(bins):
                estimated_cum = pareto.cdf(bin, kHat, loc, sigmaHat)
                pvalue_emp[b] = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

            pvalue = (pvalue_emp[pvalue_emp > 0])[-1]

        # Plot Pareto and histogram
        if plot:
            plt.figure()
            
            plt.hist(right_tail, density=True)
            x = np.linspace(np.min(right_tail), np.max(right_tail)*1.5, 100)
            pdf = pareto.pdf(x, kHat, loc, sigmaHat)
            plt.plot(x, pdf)
            # plt.xlim(0, np.max(x))

            plt.axvline(critical_value-q)
            
            figpath = 'stats_ss/pareto/'
            if not os.path.exists(figpath):
                os.makedirs(figpath)

            plt.savefig('{}{}_pareto.png'.format(figpath, plot))
    
    # If Pareto can't be estimated, get p-value based on position
    else:
        print('Impossible to estimate pareto')
        
        # Get position of true result in sorted matrix for each domain
        results_sort = np.sort(results)
        positions = (np.where(results_sort == critical_value))[0][-1]
        pvalue = 1-((positions)/(n_permutations+1))

    
    bins = np.arange(0, 5*np.max(right_tail), effect_eps/10)
    pvalue_emp = np.full((len(bins)), np.nan)
    for b, bin in enumerate(bins):
        estimated_cum = pareto.cdf(bin, kHat, loc, sigmaHat)
        pvalue_emp[b] = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))
    
    idx = (np.where(pvalue_emp > critical_p))[0][-1]
    critical_value_at_p = bins[idx]+q

    return pvalue, critical_value_at_p

def pvals(results, critical_p=0.05, pareto=False, plot=False):

    """
    Get p value for each domain
    
    Inputs:
    - results : array, 2d matrix of shape = n_perms by n_domains, containing R2 values for each permutation and each domain
    - critical_p : float, alpha level; default = 0.05
    - pareto : bool, wether to do pareto; default = False
    - plot : bool, whether to plot pareto; default = False

    Outputs:
    - pvalues : array of shape = n_domains
    - critical_values_at_p : 
    """

    # Define number of domains and permutations
    n_domains = results.shape[1]
    n_perms = results.shape[0]-1

    # Get true R
    r_true = results[0,:]

    # calculate pvals of all perms
    res_sorted = np.sort(results, axis=0)
        
    pos_all = np.array([np.searchsorted(res_sorted[:,d], results[:,d]) for d in range(n_domains)]).T
        
    pvals_all = 1-((pos_all)/(n_perms+1))

    # Pareto
    if pareto:
        # Initialize results matrices
        pvals = np.full((n_domains), np.nan)
        critical_values_at_p = np.full((n_domains), np.nan)

        for d in range(n_domains):
            if plot:
                plotname = 'ROI{}_dom{}'.format(plot, d)
            else:
                plotname = plot
            pvals[d], critical_values_at_p[d] = pareto_right(results[:,d], critical_p=critical_p, plot=plotname)

    else:
        # Sort permuted results
        res_sorted = np.sort(results, axis=0)

        # Get position of true result in sorted matrix for each domain
        positions = np.array([np.where(np.isclose(res_sorted[:,d], r_true[d]))[0][0] for d in range(n_domains)])
        
        # Calculate pval based on position
        pvals = 1-((positions)/(n_perms+1))

        # Get critical values at p
        critical_values_at_p = np.percentile(results, (1-critical_p)*100, axis=0)

    return pvals, critical_values_at_p, pvals_all

def fisher_sum(pvals_all_subs):

    n_test = pvals_all_subs.shape[-1]

    t = -2 * (np.sum(np.log(pvals_all_subs), axis=2))
    pval = 1-(chi2.cdf(t, 2*n_test))

    return pval


if __name__ == '__main__':

    rois_7 = np.unique(image.load_img('data/simulazione_datasets/sub-7/anat/ROIsem200.nii').get_fdata())
    rois_8 = np.unique(image.load_img('data/simulazione_datasets/sub-8/anat/ROIsem200.nii').get_fdata())
    rois_9 = np.unique(image.load_img('data/simulazione_datasets/sub-9/anat/ROIsem200.nii').get_fdata())

    res_7 = np.load('data/cca_results/sub-7/CCA_res_sub-7_Schaefer200.npz', allow_pickle=True)['result_dict'].item()
    res_8 = np.load('data/cca_results/sub-8/CCA_res_sub-8_Schaefer200.npz', allow_pickle=True)['result_dict'].item()
    res_9 = np.load('data/cca_results/sub-9/CCA_res_sub-9_Schaefer200.npz', allow_pickle=True)['result_dict'].item()

    n_rois = len(res_7.keys())
    n_perms = res_7[1].shape[1]
    n_doms = res_7[1].shape[2]
    n_subs = 3
    sub_list=[7,8,9]

    
    pvals_group = np.empty((n_perms, n_doms, n_rois))
    
    for roi in np.arange(1, n_rois):
        
        # Initialize
        pvals_all_subs = np.empty((n_perms, n_doms, n_subs))
        
        for s, sub in enumerate(sub_list):
            res = np.load('data/cca_results/sub-{}/CCA_res_sub-{}_Schaefer200.npz'.format(sub, sub), allow_pickle=True)['result_dict'].item()
            _, _, pvals_all_subs[:,:,s] = pvals(res[roi][1], pareto=False)
            
        # Fisher
        pvals_group[:,:, roi-1] = fisher_sum(pvals_all_subs)

    print('f')
