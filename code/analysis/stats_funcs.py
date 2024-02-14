import numpy as np
from scipy.stats import pareto, chi2
from matplotlib import pyplot as plt
import os
from nilearn import image


def plot_pareto(right_tail, kHat, loc, sigmaHat, q, critical_value, pname):
    
    x = np.linspace(np.min(right_tail), np.max(right_tail)*1.5, 100)
    pdf = pareto.pdf(x, kHat, loc, sigmaHat)
    cdf = pareto.cdf(x, kHat, loc, sigmaHat)
    pvalues = 1 - (0.9 + (cdf*(1 - 0.9)))

    estimated_cum = pareto.cdf(critical_value-q, kHat, loc, sigmaHat)
    pvalue = 1 - (0.9 + (estimated_cum*(1 - 0.9)))

    # Plot Pareto and histogram
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.spines['top'].set_visible(False)
             
    ax1.hist(right_tail+q, density=True, color='grey')
    pdf_plot, = ax1.plot(x+q, pdf, color='orange', label='pdf')
    critv = ax1.axvline(critical_value, color='red', label='critical value')
    ax1.set_ylim([0, np.max(pdf)])

    ax1.set_xlabel('R2')
    ax1.set_ylabel('Pareto pdf', color='orange')

    ax2 = ax1.twinx()
    pvals, = ax2.plot(x+q, pvalues, color='C0', label='p values')
    ax2.set_ylabel('p value', color='C0')
    ax2.spines['top'].set_visible(False)
    ax2.axhline(0.05, color='brown', ls='--', alpha=0.2)
    ax2.set_ylim([0, np.max(pvalues)])
    ax2.plot(critical_value, pvalue, marker='*', color='k')
    ax2.text(critical_value+(critical_value/500), pvalue+(pvalue/50), round(pvalue,3))
    plt.legend([pdf_plot, critv, pvals], ["Pareto pdf", "critical value", "p values"])
    

    roi, dom = pname.split('_')
    fig.suptitle('ROI {} domain {}'.format(roi, dom))

    figpath = 'data/cca_results/group/debug/pareto/'
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

def get_pval_pareto(results, tail_percentile=0.9, plot=''):

    """
    Get p value modeling the null distribution tail using Pareto
    
    Inputs:
    - results : array, 1d matrix of shape = n_perms containing R2 values for each permutation
    - tail_percentile : float, percentile of tail to model with Pareto; default = 0.9
    
    Outputs:
    - pvalue : float
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

        if plot:
            plot_pareto(right_tail, kHat, loc, sigmaHat, q, critical_value, pname=plot)
    
    # If Pareto can't be estimated, get p-value based on position
    else:
        pvalue = get_pvals(results)[0]

    return pvalue

def get_pvals_sub(sub):
    
    res_sub = np.load('data/cca_results/sub-{}/CCA_res_sub-{}_Schaefer200.npz'.format(sub, sub), allow_pickle=True)['result_dict'].item()
    pvals_sub = {}
    res_sub_dict = {}

    for roi in res_sub.keys():
        res_roi = res_sub[roi][1,:,:]
        pvals_sub[roi] = np.array([get_pvals(res_roi[:,d]) for d in range(res_roi.shape[-1])]).T
        res_sub_dict[roi] = res_roi

    return res_sub_dict, pvals_sub

def get_pvals_group(rois, pvals_subs, res_subs, n_perms, n_doms):
    
    pvals_group = {}
    results_group = {}
    
    for roi in rois:
        # Aggregate results over subjects
        pvals_aggregated = np.empty((n_perms, n_doms, len(pvals_subs.keys())))
        res_aggregated = np.empty((n_doms, len(pvals_subs.keys())))
    
        for s, sub in enumerate(pvals_subs.keys()):
            pvals_aggregated[:,:,s] = pvals_subs[sub][roi]
            res_aggregated[:,s] = res_subs[sub][roi][0,:]

        # Get mean results across subjects, for each domain
        results_group[roi] = np.mean(res_aggregated, axis=-1)
        
        # Get group p_val with Fisher's sum
        summed_pvals, _ = fisher_sum(pvals_aggregated)

        # Get non-parametric results on summed pvals (with pareto)
        pvals_group[roi] = np.array([get_pval_pareto((1-summed_pvals[:,d]), plot='{}_{}'.format(roi, d+1)) for d in range(n_doms)])

    return results_group, pvals_group

def fisher_sum(pvals_all_subs):

    n_test = pvals_all_subs.shape[-1]

    t = -2 * (np.sum(np.log(pvals_all_subs), axis=2))
    pval = 1-(chi2.cdf(t, 2*n_test))

    return pval, t

def save_nifti(atlas, n_doms, results_group, pvals_group, path):
    
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
    
    # Load Atlas
    atlas = image.load_img('../../Atlases/Schaefer-200_7Networks_ICBM152_Allin.nii.gz')
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    
    sub_list = [7,8,9]
    
    # Get pvalues for all subjects
    results_subs = {}
    pvals_subs = {}
    for s, sub in enumerate(sub_list):
        results_subs[sub], pvals_subs[sub] = get_pvals_sub(sub)

    n_perms, n_doms = 1000+1, 5

    # Get group results
    results_group, pvals_group = get_pvals_group(atlas_rois, pvals_subs, results_subs, n_perms, n_doms)

    # Save as nifti
    folder_path = 'data/cca_results/group/debug/'
    image_final = save_nifti(atlas, n_doms, results_group, pvals_group, folder_path)
    
    print('f')