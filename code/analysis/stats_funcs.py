import numpy as np
from scipy.stats import pareto, chi2
from matplotlib import pyplot as plt
import os
from nilearn import image


def plot_pareto(right_tail, kHat, loc, sigmaHat, q, critical_value, pname):
    
    """
    Plot and save right tail of null distribution, Pareto and pvalues curves
    
    Inputs:
    - right_tail : array, 1d array containing right tail of null distribution
    - kHat : float, Pareto parameter estimated
    - loc : float, Pareto parameter estimated
    - sigmaHat : float, Pareto parameter estimated 
    - q : float, value in null distribution at given percentile
    - critical_value : float, True result
    - pname : str, containing ROI number and domain number for which ti plot Pareto
    """

    # Estimate Pareto and Pvalues
    x = np.linspace(np.min(right_tail), np.max(right_tail)*1.5, 100)
    pdf = pareto.pdf(x, kHat, loc, sigmaHat)
    cdf = pareto.cdf(x, kHat, loc, sigmaHat)
    pvalues = 1 - (0.9 + (cdf*(1 - 0.9)))

    estimated_cum = pareto.cdf(critical_value-q, kHat, loc, sigmaHat)
    pvalue = 1 - (0.9 + (estimated_cum*(1 - 0.9)))

    # Create figure and axes
    plt.figure()
    fig, ax1 = plt.subplots()

    # Plot Pareto, histogram and critical value
    ax1.hist(right_tail+q, density=True, color='grey')
    pdf_plot, = ax1.plot(x+q, pdf, color='orange', label='pdf')
    critv = ax1.axvline(critical_value, color='red', label='critical value')

    # Adjust y limits, axes, labels
    ax1.set_ylim([0, np.max(pdf)])
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('R2')
    ax1.set_ylabel('Pareto pdf', color='orange')

    # Add second y axis on the right
    ax2 = ax1.twinx()

    # Plot pvalues, significance level, pvalue of critical value
    pvals, = ax2.plot(x+q, pvalues, color='C0', label='p values')
    ax2.axhline(0.05, color='brown', ls='--', alpha=0.2)
    ax2.plot(critical_value, pvalue, marker='*', color='k')
    ax2.text(critical_value+(critical_value/500), pvalue+(pvalue/50), round(pvalue,3))
    
    # Adjust y limits, axes, labels
    ax2.set_ylabel('p value', color='C0')
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim([0, np.max(pvalues)])
    
    # Add legend and title
    plt.legend([pdf_plot, critv, pvals], ["Pareto pdf", "critical value", "p values"])
    roi, dom = pname.split('_')
    fig.suptitle('ROI {} domain {}'.format(roi, dom))

    # Save figure
    figpath = 'data/cca_results/group/pareto/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    fig.savefig('{}{}_pareto.png'.format(figpath, pname))

def get_pvals(results):
    
    """
    Get p value for each permutation
    
    Inputs:
    - results : array, 1d matrix of shape = n_perms containing R2 values for each permutation

    Outputs:
    - pvals : array, 1d matrix of shape = n_perms 
    """
    # Sort results
    res_sorted = np.sort(results)

    # Calculate p-value based on position on sorted array
    pos_all = np.searchsorted(res_sorted, results)
    pvals = 1-((pos_all)/(results.shape[0]))

    return pvals

def get_pvals_sub(sub, save=True):
    
    """
    Get p values for each subject (each permutation and each domain)
    
    Inputs:
    - sub : int, sub number
    - save : bool, whether to save single subject's results; default=True

    Outputs:
    - res_sub_dict : dict, containing ROIs as keys and R2 results as values (2d array of shape = n_perms by n_doms)
    - pvals_sub : dict, containing ROIs as keys and pvals as values (2d array of shape = n_perms by n_doms)

    Calls:
    - get_pvals()
    """

    # Load R2 results of single subject
    res_sub = np.load('data/cca_results/sub-{}/CCA_res_sub-{}_Schaefer200.npz'.format(sub, sub), allow_pickle=True)['result_dict'].item()
    
    # Initialize dictionaries
    pvals_sub = {}
    res_sub_dict = {}

    # Iterate over ROIs and get R2 results and calculate p-values
    for roi in res_sub.keys():
        res_roi = res_sub[roi][1,:,:]
        pvals_sub[roi] = np.array([get_pvals(res_roi[:,d]) for d in range(res_roi.shape[-1])]).T
        res_sub_dict[roi] = res_roi

    # Save results
    if save:
        path = 'data/cca_results/sub-{}/'.format(sub)
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez(path + 'CCA_stats_sub-{}'.format(sub), pvals_sub=pvals_sub, res_sub_dict=res_sub_dict)  

    return res_sub_dict, pvals_sub

def get_pval_pareto(results, tail_percentile=0.9, plot=None):

    """
    Get p value modeling the null distribution tail using Pareto
    
    Inputs:
    - results : array, 1d matrix of shape = n_perms containing R2 values for each permutation
    - tail_percentile : float, percentile of tail to model with Pareto; default = 0.9
    - plot : str, ROI and domain name for wich to plot and save pareto; default = None, that is don't plot
    
    Outputs:
    - pvalue : float, p value of getting true R by chance given null distribution

    Calls:
    - plot_pareto()
    - get_pvals()
    """

    # Define critical value, null distribution and tail
    null_distro = results[1:]
    critical_value = results[0]
    n_permutations = null_distro.shape[0]
    q = np.percentile(null_distro, tail_percentile*100)
    right_tail = null_distro[null_distro>q] - q # Recentered
    
    # Fit pareto over right tail and estimate Pareto distro parameters
    kHat, loc, sigmaHat = pareto.fit(right_tail, floc=0)

    # Verify that Pareto can be estimated (if kHat < 0.5 Pareto might not be a good approssimation of the tail), and that the critical value is in the tail
    if kHat > -0.5 and (critical_value-q) > 0:
        
        # Estimate CDF of Pareto distro at critical value and calculate p-value
        estimated_cum = pareto.cdf(critical_value-q, kHat, loc, sigmaHat)
        pvalue = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

        # Verify that the calculated p-value is not too small (possibilità di problemi numerici o di approssimazione nella stima della distribuzione di Pareto stessa)
        if pvalue < (np.finfo(np.float64).eps/2):
            
            # Calculate step size as a fraction of the maximum value within the right tail
            effect_eps = np.max(right_tail)/(n_permutations/len(right_tail)) 
            
            # Define bins going from 0 to critical value(recentered) in steps
            bins = np.arange(0, critical_value-q, effect_eps)
            
            # Calculate p-values for each bin 
            pvalue_emp = np.full((len(bins)), np.nan)
            for b, bin in enumerate(bins):
                estimated_cum = pareto.cdf(bin, kHat, loc, sigmaHat)
                pvalue_emp[b] = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

            # Select highest p-value across bins
            pvalue = (pvalue_emp[pvalue_emp > 0])[-1]

        # Plot Pareto
        if plot:
            plot_pareto(right_tail, kHat, loc, sigmaHat, q, critical_value, pname=plot)
    
    # If Pareto can't be estimated, get p-value based on position
    else:
        pvalue = get_pvals(results)[0]

    return pvalue

def fisher_sum(pvals_all_subs):
    
    """
    Compute Fisher sum of pvalues across subjects
    
    Inputs:
    - pvals_all_subs : array, 3d matrix of shape = n_perms by n_doms by n_subs, containing pvals of each subject, domain and perm
    
    Outputs:
    - pval : array, 2d matrix of shape = n_perms by n_doms, containing aggregated pvalues
    - t : array, 2d matrix of shape = n_perms by n_doms, containing aggregated t statistic
    """
    
    # Define number of tests, in this case = number of subjects
    n_test = pvals_all_subs.shape[-1]

    # Calculate t and pval with Fisher's formula
    t = -2 * (np.sum(np.log(pvals_all_subs), axis=2))
    pval = 1-(chi2.cdf(t, 2*n_test))

    return pval, t

def get_pvals_group(rois, pvals_subs, res_subs, n_perms, n_doms, save=True):
    
    """
    Get p values at group level (each permutation and each domain)
    
    Inputs:
    - rois : list or array of ints or strings, list of ROIs (keys of single subject results)
    - pvals_subs : dict, containing sub number as keys and dictionaries as values, containing ROIs as keys and pvals as values (2d array of shape = n_perms by n_doms)
    - res_subs : dict, containing sub number as keys and dictionaries as values, containing ROIs as keys and R2 as values (2d array of shape = n_perms by n_doms)
    - n_perms : int, number of permutations
    - n_doms : int, number of domains
    - save : bool, whether to save group results; default=True

    Outputs:
    - results_group : dict, containing ROIs as keys and mean R2 results of non permuted data as values (1d array of shape = n_doms)
    - pvals_group : dict, containing ROIs as keys and pvals as values (2d array of shape = n_perms by n_doms)

    Calls:
    - fisher_sum()
    - get_pval_pareto()
    """
    
    # Initialize dictionaries
    pvals_group = {}
    results_group = {}
    
    # Iterate over ROIs
    for roi in rois:

        # Initialize arrays for aggregated results
        pvals_aggregated = np.empty((n_perms, n_doms, len(pvals_subs.keys())))
        res_aggregated = np.empty((n_doms, len(pvals_subs.keys())))

        # Aggregate results over subjects
        for s, sub in enumerate(pvals_subs.keys()):
            pvals_aggregated[:,:,s] = pvals_subs[sub][roi]
            res_aggregated[:,s] = res_subs[sub][roi][0,:]

        # Get mean results across subjects, for each domain
        results_group[roi] = np.mean(res_aggregated, axis=-1)
        
        # Get group p_val with Fisher's sum
        summed_pvals, _ = fisher_sum(pvals_aggregated)

        # Get non-parametric results on summed pvals (with pareto)
        pvals_group[roi] = np.array([get_pval_pareto((1-summed_pvals[:,d]), plot='{}_{}'.format(roi, d+1)) for d in range(n_doms)])

    # Save results
    if save:
        path = 'data/cca_results/group/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez(path + 'CCA_stats_group', pvals_group=pvals_group, results_group=results_group)  

    return results_group, pvals_group

def save_nifti(atlas, n_doms, results_group, pvals_group, path):
    
    """
    Create and save Nifti image of results
    
    Inputs:
    - atlas : nii like object, Atlas in nifti
    - n_doms : int, number of domains
    - results_group : dict, containing ROIs as keys and mean R2 results of non permuted data as values (1d array of shape = n_doms)
    - pvals_group : dict, containing ROIs as keys and pvals as values (2d array of shape = n_perms by n_doms)
    - path : str, path where to save results

    Outputs:
    - image_final : array, 4d matrix of shape = x by y by z by n_doms*2, containing mean results for each domain and aggregated pvalues for each domain
    """

    # Get atlas dimensions
    atlas_data = atlas.get_fdata()
    atlas_rois = np.unique(atlas_data).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    x,y,z = atlas_data.shape

    # Initialize image matrix
    image_final = np.zeros((x, y, z, n_doms*2))

    # Assign group R2 and p_value to each voxel
    for roi in atlas_rois:
        x_inds, y_inds, z_inds = np.where(atlas_data==roi)
        
        image_final[x_inds, y_inds, z_inds, :n_doms] = results_group[roi]
        image_final[x_inds, y_inds, z_inds, n_doms:] = 1-pvals_group[roi]
    
    # Save
    if not os.path.exists(path):
        os.makedirs(path)

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename('{}cca_results_group.nii'.format(path))

    return image_final
    
if __name__ == '__main__':
    
    print('Starting statistical analyses')
    
    n_perms, n_doms, sub_list = 1000, 5, [7,8,9]

    # Load Atlas
    atlas = image.load_img('../../Atlases/Schaefer-200_7Networks_ICBM152_Allin.nii.gz')
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    
    # Get pvalues for all subjects
    results_subs = {}
    pvals_subs = {}
    for s, sub in enumerate(sub_list):
        results_subs[sub], pvals_subs[sub] = get_pvals_sub(sub, save=True)

    # Get group results
    results_group, pvals_group = get_pvals_group(atlas_rois, pvals_subs, results_subs, n_perms+1, n_doms, save=True)

    # Save as nifti
    folder_path = 'data/cca_results/group/'
    image_final = save_nifti(atlas, n_doms, results_group, pvals_group, folder_path)

    print('Finished statistical analyses')