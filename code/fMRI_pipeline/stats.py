import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from nilearn import image
from scipy.stats import pareto, chi2
from matplotlib import pyplot as plt
from scipy.stats import false_discovery_control as fdr

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
    figpath = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/group/pareto/'
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

def get_pvals_sub(condition, sub, adjusted=True, save=True, suffix='', atlas_file='Schaefer200', global_path=None, save_nifti_opt=False):

    """
    Get p values for each subject (each permutation and each domain)
    
    Inputs:
    - condition : str, indicates condition, i.e. fMRI session the subjects were assigned to
    - sub : str, sub number
    - adjusted : bool, whether to get adjusted R2 or not adjusted R2; default=True (get adjusted)
    - save : bool, whether to save single subject's results; default=True
    - suffix : str, default=''
    - atlas_file : str, default 'Schaefer200'
    - global_path : str, default = None
    - save_nifti_opt : bool, whether to create nifti with results or not; default=False

    Outputs:
    - res_sub_mat : array, containing R2 results for each ROI, perm and domain (3d array of shape = n_rois by n_perms by n_doms)
    - pvals_sub : array, containing pvalues for each ROI, perm and domain (3d array of shape = n_rois by n_perms by n_doms)
    - rois : list, containing ROIs

    Calls:
    - get_pvals()
    - save_nifti()
    """

    if global_path is None: global_path = os.getcwd()

    # Load R2 results of single subject
    res_sub = np.load('{}cca_results/{}/sub-{}/{}/CCA_res_sub-{}_{}.npz'.format(global_path, condition, sub, suffix, sub, atlas_file), allow_pickle=True)['result_dict'].item()
    rois = list(res_sub.keys())
    n_rois = len(rois)
    n_perms = res_sub[rois[0]].shape[1]
    n_doms = res_sub[rois[0]].shape[2]

    # Initialize results matrices and dictionaries
    res_sub_mat = np.full((n_rois, n_perms, n_doms), np.nan)
    pvals_sub_mat = np.full((n_rois, n_perms, n_doms), np.nan)
    res_sub_dict = {}
    pvals_sub_dict = {}

    rind = 1 if adjusted else 0
    
    # Iterate over ROIs and get R2 results and calculate p-values
    for r, roi in enumerate(rois):
        res_roi = res_sub[roi][rind,:,:] # shape n_perms by n_doms 
        pvals_roi = np.array([get_pvals(res_roi[:,d]) for d in range(res_roi.shape[-1])]).T
        
        # Assign to results matrix and dict
        res_sub_mat[r] = res_roi
        pvals_sub_mat[r] = pvals_roi
        res_sub_dict[roi] = res_roi[0]
        pvals_sub_dict[roi] = pvals_roi[0]

    # Save results
    if save:
        path = '{}cca_results/{}/sub-{}/{}/'.format(global_path, condition, sub, suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez(path + 'CCA_R2{}_pvals_sub-{}_{}'.format('adj' if adjusted else '', sub, atlas_file),  res_sub_mat=res_sub_mat, pvals_sub_mat=pvals_sub_mat, rois_order=rois)  

    # Create nifti
    if save_nifti_opt:
        path = '{}cca_results/{}/sub-{}/{}/'.format(global_path, condition, sub, suffix)
        if not os.path.exists(path):
            os.makedirs(path)

        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
        image_final = save_nifti(atlas, n_doms, res_sub_dict, pvals_sub_dict, rois, path+'CCA_R2{}_pvals_sub-{}_{}'.format('adj' if adjusted else '', sub, atlas_file))

    return res_sub_mat, pvals_sub_mat, rois

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

def get_pvals_group(condition, rois, pvals_subs, res_subs, maxT=False, FDR=False, save=True, suffix='', atlas_file='Schaefer200', global_path=None, save_nifti_opt=False):

    """
    Get p values at group level (each permutation and each domain)
    
    Inputs:
    - condition : str, indicates condition, i.e. fMRI session the subjects were assigned to
    - rois : list or array of ints or strings, list of ROIs (keys of single subject results)
    - pvals_subs : array, 4d array of shape = n_subs by n_rois by n_perms by n_doms
    - res_subs : array, 4d array of shape = n_subs by n_rois by n_perms by n_doms
    - maxT : bool, whether to correct for multiple comparisons using max-T correction; default=False
    - FDR : bool, whether to correct for multiple comparisons using FDR correction; default=False
    - save : bool, whether to save group results; default=True
    - suffix : str, suffix; default = ''
    - atlas_file : str, default 'Schaefer200'
    - global_path : str, default = None
    - save_nifti_opt : bool, whether to create nifti with results or not; default=False

    Outputs:
    - results_group : dict, containing ROIs as keys and mean R2 results of non permuted data as values (1d array of shape = n_doms)
    - pvals_group : dict, containing ROIs as keys and pvals as values (2d array of shape = n_perms by n_doms)

    Calls:
    - fisher_sum()
    - get_pval_pareto()
    """
    
    if global_path is None: global_path = os.getcwd()

    # Get dimensions
    n_subs, n_rois, n_perms, n_doms = res_subs.shape
    
    # Aggregate results across subs by averaging real r
    results_group = np.mean(res_subs, axis=0)[:,0,:]
    results_group_dict = {roi: results_group[r,:] for r, roi in enumerate(rois)}
    
    # Initialize pvals matrix and dict
    aggregated_pvals = np.empty((n_rois, n_perms, n_doms))
    aggregated_stats = np.empty((n_rois, n_perms, n_doms))
    pvals_group = np.empty((n_rois, n_doms))
    pvals_group_dict = {}

    # Iterate over ROIs
    for r, roi in enumerate(rois):
        pvals_roi = np.rollaxis(pvals_subs, 0, 4)[r]
        
        # Get group p_val with Fisher's sum
        aggregated_pvals[r], aggregated_stats[r] = fisher_sum(pvals_roi)

    # Max-T correction: get max distro nulla across rois
    if maxT:
        distro_maxt = np.max(np.max(aggregated_stats, axis=-1), axis=0)

    # Get non-parametric results on summed pvals (with pareto)
    for r, roi in enumerate(rois):
        for d in range(n_doms):
            if maxT:
                pvals_roi_dom = np.append(aggregated_stats[r,0,d], distro_maxt[1:])

            else:
                pvals_roi_dom = aggregated_stats[r,:,d]

            pvals_group[r,d] = np.array(get_pval_pareto(pvals_roi_dom))
        pvals_group_dict[roi] = pvals_group[r,:]

    # FDR correction
    if FDR:
        pvals_group = np.reshape(fdr(np.ravel(pvals_group)), pvals_group.shape)

    # Save results
    if save:
        path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez('{}CCA_R2_pvals{}{}_group_{}'.format(path, '_maxT' if maxT else '', '_FDR' if FDR else '', atlas_file), pvals_group=pvals_group, results_group=results_group, rois_list=rois)  

    # Create nifti
    if save_nifti_opt:
        path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
        if not os.path.exists(path):
            os.makedirs(path)

        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
        image_final = save_nifti(atlas, n_doms, results_group_dict, pvals_group_dict, rois, '{}CCA_R2_pvals{}_group_{}'.format(path,'_maxT' if maxT else '', atlas_file))

    return results_group, pvals_group

def save_nifti(atlas, n_doms, results, pvals={}, rois_to_include=[], path=''):

    """
    Create and save Nifti image of results
    
    Inputs:
    - atlas : nii like object, Atlas in nifti
    - n_doms : int, number of domains
    - results : dict, containing ROIs as keys and R2 results of non permuted data as values (1d array of shape = n_doms)
    - pvals : dict, containing ROIs as keys and pvals as values (1d array of shape = n_doms); default = {} i.e. no pvals
    - rois_to_include : list, list of ROIs to be considered; default = [], i.e. include all atlas ROIs
    - path : str, path where to save results including nifti filename

    Outputs:
    - image_final : array, 4d matrix of shape = x by y by z by n_doms*2, containing results for each domain and pvalues for each domain
    """

    # Get atlas dimensions
    atlas_data = atlas.get_fdata()
    atlas_rois = np.unique(atlas_data).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    x,y,z = atlas_data.shape

    # If indicated, select only specified ROIs 
    if len(rois_to_include)<=0:
        rois_to_include = list(atlas_rois)
    
    rois = np.array([r for r in rois_to_include if r in list(atlas_rois)])

    # Initialize image matrix
    if len(pvals)>0:
        image_final = np.zeros((x, y, z, n_doms*2))
    else:
        image_final = np.zeros((x, y, z, n_doms))

    # Assign group R2 and p_value to each voxel
    for roi in rois:
        x_inds, y_inds, z_inds = np.where(atlas_data==roi)
        
        image_final[x_inds, y_inds, z_inds, :n_doms] = results[roi]
        
        if len(pvals)>0:
            image_final[x_inds, y_inds, z_inds, n_doms:] = 1-pvals[roi]
    
    # Save
    if not os.path.exists(''.join([x + '/' for x in (path.split('/')[:-1])])):
        os.makedirs(''.join([x + '/' for x in (path.split('/')[:-1])]))

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename(path)

    return image_final

def get_results_sub(condition, sub, adjusted=True, save=True, suffix='', atlas_file='Schaefer200', global_path=None, save_nifti_opt=False):
    
    """
    Get p values for each subject (each permutation and each domain)
    
    Inputs:
    - condition : str, indicates condition, i.e. fMRI session the subjects were assigned to
    - sub : str, sub number
    - adjusted : bool, whether to get adjusted R2 or not adjusted R2; default=True (get adjusted)
    - save : bool, whether to save single subject's results; default=True
    - suffix : str, default=''
    - atlas_file : str, default 'Schaefer200'
    - global_path : str, default = None
    - save_nifti_opt : bool, whether to create nifti with results or not; default=False

    Outputs:
    - res_sub_mat : array, containing R2 results for each ROI, perm and domain (3d array of shape = n_rois by n_perms by n_doms)    
    - rois : list, containing ROIs

    Calls:
    - get_pvals()
    - save_nifti()
    """

    if global_path is None: global_path = os.getcwd()

    # Load R2 results of single subject
    res_sub = np.load('{}cca_results/{}/sub-{}/{}/CCA_res_sub-{}_{}.npz'.format(global_path, condition, sub, suffix, sub, atlas_file), allow_pickle=True)['result_dict'].item()
    rois = list(res_sub.keys())
    n_rois = len(rois)
    n_perms = res_sub[rois[0]].shape[1]
    n_doms = res_sub[rois[0]].shape[2]

    # Initialize results matrix and dictionary
    res_sub_mat = np.full((n_rois, n_perms, n_doms), np.nan)
    res_sub_dict = {}
    rind = 1 if adjusted else 0
    
    # Iterate over ROIs and get R2 results
    for r, roi in enumerate(rois):
        res_roi = res_sub[roi][rind,:,:] # shape n_perms by n_doms 
      
        # Assign to results matrix and dict
        res_sub_mat[r] = res_roi
        res_sub_dict[roi] = res_roi[0]

    # Save results
    if save:
        path = '{}cca_results/{}/sub-{}/{}/'.format(global_path, condition, sub, suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez(path + 'CCA_R2{}_sub-{}_{}'.format('adj' if adjusted else '', sub, atlas_file),  res_sub_mat=np.squeeze(res_sub_mat), rois_order=rois)  

    # Create nifti
    if save_nifti_opt:
        path = '{}cca_results/{}/sub-{}/{}/'.format(global_path, condition, sub, suffix)
        if not os.path.exists(path):
            os.makedirs(path)

        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
        image_final = save_nifti(atlas, n_doms, res_sub_dict, {}, rois, path+'CCA_R2{}_sub-{}_{}'.format('adj' if adjusted else '', sub, atlas_file))

    return res_sub_mat, rois

if __name__ == '__main__': 
    
    # Set options and parameters
    condition = 'aud'
    full_model_opt = False # full_model vs variance_partitioning (if False run Variance partitioning)

    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    sub_list = sub_lists[condition]
    n_subs = len(sub_list)
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    n_perms = 1000 if full_model_opt else 0
    atlas_file = 'Schaefer200'
    rois_to_include = list(np.loadtxt('{}cca_results/AV/group/fullmodel/significantROIs_AV.txt'.format(global_path)).astype(int)) if condition != 'AV' else []
    suffix = 'fullmodel' if full_model_opt else 'variancepart'

    ss_stats = True
    adjusted = False

    group_stats = False
    maxT = True
    FDR = False

    save = True
    save_stats_nifti = False

    # Load task and Create Full model
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    full_model = {'full_model': np.hstack([domains[d] for d in domains_list])}
    if full_model_opt:
        domains = full_model

    n_doms = len(domains.keys())

    # Load Atlas
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))
    x,y,z = atlas.get_fdata().shape

    # If indicated, select only specified ROIs 
    if len(rois_to_include)<=0:
        rois_to_include = atlas_rois
    
    rois = np.array([r for r in rois_to_include if r in atlas_rois])
    n_rois = len(rois)

    # Single subject stats
    if ss_stats:
        print('Starting single-sub statistical analyses')

        # Get pvalues for all subjects
        rois_list = np.full((len(sub_list), n_rois), np.nan)
        results_subs = np.full((len(sub_list), n_rois, n_perms+1, n_doms), np.nan)

        if n_perms>0:
            pvals_subs = np.full((len(sub_list), n_rois, n_perms+1, n_doms), np.nan)

            for s, sub in enumerate(sub_list):
                sub_str = str(sub) if len(str(sub))>=2 else '0'+str(sub)
                results_subs[s], pvals_subs[s], rois_list[s] = get_pvals_sub(condition, sub_str, adjusted=adjusted, save=save, suffix=suffix, atlas_file=atlas_file, global_path=global_path, save_nifti_opt=save_stats_nifti)
                        
            # Verify that ROIs list is the same for all subjects (i.e. that no ROI is absent in any subject)
            if np.any(np.diff(rois_list, axis=0)):
                print('ROI list is not the same in every subject, check ROIs singularly')
            
            else:
                rois_list = rois_list[0].astype(int)
            
            # Save results and uncorrected pvals to numpy
            path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            np.savez('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), results=results_subs, pvals=pvals_subs, rois_list=rois_list)

        else:
            for s, sub in enumerate(sub_list):
                sub_str = str(sub) if len(str(sub))>=2 else '0'+str(sub)
                results_subs[s], rois_list[s] = get_results_sub(condition, sub_str, adjusted=adjusted, save=save, suffix=suffix, atlas_file=atlas_file, global_path=global_path, save_nifti_opt=save_stats_nifti)
            
             # Verify that ROIs list is the same for all subjects (i.e. that no ROI is absent in any subject)
            if np.any(np.diff(rois_list, axis=0)):
                print('ROI list is not the same in every subject, check ROIs singularly')
            
            else:
                rois_list = rois_list[0].astype(int)
            
            # Save results and uncorrected pvals to numpy
            path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            np.savez('{}CCA_R2{}_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), results=results_subs, rois_list=rois_list)

        print('Finished single-sub statistical analyses')


    # Group level stats
    if group_stats:
        print('Starting group statistical analyses')
        
        # Load single subject results
        path = '{}cca_results/{}/group/{}/'.format(global_path, condition, suffix)
        results = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['results']
        rois = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['rois_list']
        
        if n_perms > 0:
            # Load single subject pvals
            pvals = np.load('{}CCA_R2{}_pvals_allsubs_{}.npz'.format(path, 'adj' if adjusted else '', atlas_file), allow_pickle=True)['pvals']
        
            # Get aggregated results and pvals
            results_group, pvals_group = get_pvals_group(condition, rois, pvals, results, maxT=maxT, FDR=FDR, save=save, suffix=suffix, global_path=global_path, save_nifti_opt=save_stats_nifti)
        
        print('Finished group statistical analyses')
