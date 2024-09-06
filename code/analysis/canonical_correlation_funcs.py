import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import numpy as np
import multiprocessing as mp
from utils.exectime_decor import timeit
from nilearn import image
import sys
from datetime import datetime
from scipy.stats import zscore

from permutation_schema_func import permutation_schema

def pca_single_roi(roi, n_comps=None, zscore_opt=False):
    
    """
    Perform PCA of a single ROI

    Input:
    - roi : array, 2d matrix of shape = n_tpoints by n_voxels containing fMRI data 
    - n_comps : int, number of components to keep, if None, keep all components; default = None
    - zscore_opt : bool, whether to z-score components after PCA or not; default = False

    Output:
    - roi_pca : array, 2d matrix of shape = n_tpoints by n_components containing fMRI data projected on the components
    - explained_var : array, cumulative sum of percentage of explained variance by each component
    """

    # PCA
    pca = PCA(n_components=n_comps)
    roi_pca = pca.fit_transform(roi)
    explained_var = np.cumsum(pca.explained_variance_ratio_)

    # zscore
    if zscore_opt:
        roi_pca = zscore(roi_pca, axis=0)

    return roi_pca, explained_var

def canonical_correlation(X,Y, center=True):
    
    """
    Performs canonical correlation analysis using sklearn.cross_decomposition CCA package
    
    Inputs:
    - X : array, 2d matrix of shape = n by d1
    - Y : array, 2d matrix of shape = n by d2
    - center: bool, whether to remove the mean (columnwise) from each column of X and Y; default = True

    Outputs:
    - r2 : float, R-squared (Y*B = V)
    - r2adj : float, R-squared (Y*B = V) adjusted
    - A : Sample canonical coefficients for X variables
    - B : Sample canonical coefficients for Y variables
    - r : Sample canonical correlations
    - U : canonical scores for X
    - V : canonical scores for Y
    """

    # Center X and Y
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

    # Canonical Correlation Analysis
    n_components = np.min([X.shape[1], Y.shape[1]]) # Define n_components as the min rank

    cca = CCA(n_components=n_components, scale=True, max_iter=5000)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)

    # Get A and B matrices as done by Matlab canoncorr()
    A = np.linalg.lstsq(X, U, rcond=None)[0]
    B = np.linalg.lstsq(Y, V, rcond=None)[0]

    # Calculate R for each canonical variate
    R = np.full(U.shape[1], np.nan)
    for c in range(U.shape[1]):
        x = U[:,c]
        y = V[:,c]
        r = np.corrcoef(x,y, rowvar=False)[0,1]
        R[c] = r

    # Calculate regression coefficients b_coeffs
    b_coeffs = np.linalg.lstsq(U, V, rcond=None)[0]

    # Predict V using U and b_coeffs
    V_pred = np.dot(U, b_coeffs)

    # Calculate centered predicted Y
    Y_pred = np.dot(V_pred, np.linalg.pinv(B))

    # Calculate R-squared
    SSres = np.sum((Y.ravel() - Y_pred.ravel()) ** 2)
    SStot = np.sum((Y.ravel() - np.mean(Y.ravel())) ** 2)
    r2 = 1 - (SSres / SStot)

    # Adjust by number of X columns
    n = Y_pred.shape[0]
    p = n_components
    r2adj = 1 - (1-r2)*((n-1)/(n-p-1))

    return r2, r2adj, A, B, R, U, V

def run_cca_single_roi(roi, perm_schema, domains, variance_part=0):
    
    """
    Run canonical correlation and store R2 between fMRI activity of a ROI and each domain model, first using real data and then permuted fMRI data
    
    Inputs:
    - roi : array, 2d matrix of shape = n_tpoints by n_voxels containing fMRI data 
    - perm_schema : array, 2d matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - domains : dict, including domains as keys and 2d matrix of shape = n_tpoints by n_columns as values
    - variance_part : int, n_perms to run variance partitioning (shuffling colums of single domain); default = 0, i.e. run single domain model
    
    Outputs:
    - results : array, 3d matrix of shape = 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each permutation and each domain
        
    Calls:
    - canonical_correlation()
    - permutation_schema()
    """
    
    # Get number of permutations and number of domains
    n_perms = perm_schema.shape[1]
    n_domains = len(domains)
    
    # Initialize results matrix
    results = np.full((2, n_perms, n_domains), np.nan)
    
    # Run canonical correlation analysis for each permutation 
    for perm in range(n_perms):
        
        # Shuffle fmri signal based on permutation schema
        order = perm_schema[:, perm]
        Y = roi[order, :]
        
        # Run cca for each domain
        for d, domain in enumerate(domains):
            
            if variance_part:
                # Create permutation schema to shuffle columns of left-out domain for variance partitioning
                vperm_schema = permutation_schema(domains[domain].shape[0], n_perms=variance_part, chunk_size=1)
                
                # Initialize arrays to save all permuted results 
                r2_p = np.full(variance_part, np.nan)
                r2adj_p = np.full(variance_part, np.nan)

                # For each permutation, build full model and shuffle columns of left-out domain
                for vperm in range(1, vperm_schema.shape[1]):
                    vorder = vperm_schema[:, vperm] # first colum contains original order (non-permuted)                 
                    X = np.hstack([domains[dom] if dom!=domain else domains[dom][vorder,:] for dom in domains.keys()])
                    
                    # Run cca 
                    r2_p[vperm-1], r2adj_p[vperm-1], _, _, _, _, _ = canonical_correlation(X, Y)
                
                # Get average R2 results across perms
                r2 = np.mean(r2_p)
                r2adj = np.mean(r2adj_p)

            else:
                # Select single domain as model
                X = domains[domain]
                
                # Run cca
                r2, r2adj, _, _, _, _, _ = canonical_correlation(X, Y)
            
            results[0, perm, d] = r2
            results[1, perm, d] = r2adj

    return results

def extract_roi(data, atlas, rois_to_include=[]):
    
    """
    Parcellize data into ROIs of given atlas
    
    Inputs:
    - data : array, 4d (or 3d) matrix of shape = x by y by z (by time) containing fMRI signal
    - atlas : array, 3d matrix of shape = x by y by z containing ROIs membership for each voxel
    - rois_to_include : list, list of ROIs to be considered; default = [], i.e. include all atlas ROIs

    Outputs:
    - data_rois : dict, including for each ROI within atlas, 2d (or 1d) matrix of shape = voxels (by time) containing fMRI signal of each voxel within ROI
    - n_rois : int, number of ROIs in atlas
    - n_voxels_rois : dict, including for each ROI within atlas, the number of voxels within ROI
    """
    
    # Get list of ROIs in atlas
    rois_atlas = np.unique(atlas)
    rois_atlas = np.delete(rois_atlas, np.argwhere(rois_atlas==0))

    # If indicated, select only specified ROIs 
    if len(rois_to_include)<=0:
        rois_to_include = list(rois_atlas)
    
    rois = np.array([r for r in rois_to_include if r in list(rois_atlas)])
    n_rois = len(rois)

    # Initialize dictionaries
    data_rois = {}
    n_voxels_rois = {}
    
    # Iterate over ROIs
    for roi in rois:
        # Get coordinates of voxel within ROI
        (x,y,z) = np.where(atlas==roi)

        # Save ROI data matrix of shape n_timepoints by n_voxels and relative number of voxels in dictionary 
        data_rois[int(roi)] = data[x,y,z].T
        n_voxels_rois[int(roi)] = len(x)

    return data_rois, n_rois, n_voxels_rois

@timeit
def run_cca_all_rois(s, data_rois, domains, perm_schema, minvox, pooln=20, zscore_opt=False, skip_roi=True, variance_part=0):
    
    """
    Run canonical correlation for all ROIs using parallelization
    
    Inputs:
    - s : int, sub index
    - data_rois : dict, including for each ROI within atlas, 2d (or 1d) matrix of shape = voxels (by time) containing fMRI signal of each voxel within ROI
    - domains : dict, including domains as keys and 2d matrix of shape = n_tpoints by n_columns as values
    - perm_schema : array, 2d matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - minvox : int, number of components to keep in PCA, in this case it will be the number of voxels of the smallest ROI (provided that it is higher than the maximum number of predictors of the task domains)
    - pooln : int, number of parallelization processes; default = 20
    - zscore_opt : bool, whether to z-score components after PCA or not; default = False
    - skip_roi : bool, how to deal with ROIs with n_voxels < n_predictors, if False add missing voxels using ROI mean signal, if True skip ROI; default = True
    - variance_part : int, n_perms to run variance partitioning (shuffling colums of single domain); default = 0, i.e. run single domain model
    
    Outputs:
    - result_matrix : array, 4d matrix of shape = n_rois by 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each ROI, each permutation and each domain
    - result_dict : dict, including ROI as keys and result matrix of each ROI as values
    
    Calls:
    - run_cca_single_roi()
    - pca_single_roi()
    """

    # Define number of ROIs and of permutations
    n_rois = len(data_rois.keys())
    n_perms = perm_schema.shape[1]
    
    # Initialize results matrix and dictionary
    result_matrix = np.full((n_rois, 2, n_perms, len(domains)), np.nan)
    result_dict = {}
    pca_dict = {}

    # Set pool
    results_pool = []
    pool = mp.Pool(pooln)

    # Iterate over ROIs
    for r, roi in data_rois.items():
        
        # Verify minimum number of voxels for each ROI
        if roi.shape[1] < minvox:
            
            if skip_roi:
                print('- ROI {} voxels are {}, less than the max number of predictors!! This ROI will be discarded'.format(r, roi.shape[1]))
                continue
            
            else:
                # Add n voxels to reach min dimension by using ROI mean signal
                voxelstoadd = minvox - roi.shape[1]
                print('- ROI {} voxels are {}, less than the max number of predictors!! {} voxels added with mean ROI signal as signal'.format(r, roi.shape[1], voxelstoadd))
                
                meanvox = np.mean(roi, axis=1)
                voxelstoaddmat = np.tile(meanvox, (voxelstoadd, 1)).T
                roi = np.hstack((roi, voxelstoaddmat))
        
        # PCA
        roi_pca, pca_dict[r] = pca_single_roi(roi, n_comps=minvox, zscore_opt=zscore_opt)

        # Run canoncorr for each ROI with parallelization and store results
        result_pool = pool.apply_async(run_cca_single_roi, args=(roi_pca, perm_schema, domains, variance_part))
        results_pool.append((r, result_pool))

    pool.close()
    
    # Unpack results
    for result_pool in results_pool:
        njob = result_pool[1]._job - (s*n_rois)
        roi_n = result_pool[0]
        result_matrix[njob, :, :, :] = result_pool[1].get()
        result_dict[roi_n] = result_pool[1].get()

    return result_matrix, result_dict, pca_dict

def run_cca_all_subjects(condition, sub_list, domains, atlas_file, rois_to_include=[], n_perms=1000, chunk_size=15, seed=0, pooln=20, zscore_opt=False, skip_roi=True, variance_part=0, save=True, suffix=''):
    
    """
    Run canonical correlation for all subjects
    
    Inputs:
    - condition: str, indicates condition, i.e. fMRI session the subjects were assigned to
    - sub_list : array, 1d array containing sub numbers for which to run CCA
    - domains : dict, including domain names as keys and domain regressors as values
    - atlas_file : str, filename of atlas to use
    - rois_to_include : list, list of ROIs to be considered; default = [], i.e. include all atlas ROIs
    - n_perms : int, number of permutations (columns); default = 1000
    - chunk_size: int, size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: int, seed for the random permutation; default = 0
    - pooln : int, number of parallelization processes; default = 20 
    - zscore_opt : bool, whether to z-score components after PCA or not; default = False
    - skip_roi : bool, how to deal with ROIs with n_voxels < n_predictors, if False add missing voxels using ROI mean signal, if True skip ROI; default = True
    - variance_part : int, n_perms to run variance partitioning (shuffling colums of single domain); default = 0, i.e. run single domain model
    - save : bool, whether to save results as npy files; default = True
    - suffix : str, foldername suffix to add to saving path; default = ''
    
    Saves:
    For each subject
    - result_matrix : array, 4d matrix of shape = n_rois by 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each ROI, each permutation and each domain
    - result_dict : dict, including ROI as keys and result matrix of each ROI as values
    
    Calls:
    - extract_roi()
    - permutation_schema()
    - run_cca_all_rois()
    """

    # Set printing settings
    orig_stdout = sys.stdout
    
    # Iterate over subjects
    for s, sub in enumerate(sub_list):
        
        # Print output to txt file
        log_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/logs/'.format(sub, suffix)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        logfile = open('{}log_cca_sub-{}_{}.txt'.format(log_path, sub, atlas_file), 'w')
        sys.stdout = logfile

        print(datetime.now())
        print('CCA of sub-{}'.format(sub))
        print('- n_perms: {}'.format(n_perms))
        print('- seed: {}'.format(seed))
        print('\n- Atlas: {}'.format(atlas_file))

        # Load data
        data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-{}/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub, condition)).get_fdata()
        atlas = image.load_img('/data1/Action_teresi/CCA/atlas/sub-{}_{}_atlas_2orig.nii.gz'.format(sub, atlas_file)).get_fdata()
        
        # Extract rois
        data_rois, n_rois, n_voxels = extract_roi(data, atlas, rois_to_include)

        # Establish number of voxels minimum for ROIs and n_components for pca
        if variance_part:
            minvox = np.sum([v.shape[1] for v in domains.values()])
        else:
            minvox = np.max([v.shape[1] for v in domains.values()])

        print('- n_rois: {}'.format(n_rois))
        print('- n_voxels between {} and {}'.format(np.min(list(n_voxels.values())), np.max(list(n_voxels.values()))))
        print('- minvox used for PCA = {}'.format(minvox))
        print('- zscoring after PCA: {}'.format('on' if zscore_opt else 'off'))

        # Generate permutation schema
        n_tpoints = data.shape[-1]
        perm_schema = permutation_schema(n_tpoints, n_perms=n_perms, chunk_size=chunk_size)
    
        # Run cca for each roi
        print('\n- {} domains used for CCA: {}'.format(len(list(domains.keys())), list(domains.keys())))
        print('- Used {} perms for variance partitioning'.format(variance_part))
        print('\n- {} CPUs used'.format(pooln))

        result_matrix, result_dict, pca_dict = run_cca_all_rois(s, data_rois, domains, perm_schema, minvox, pooln=pooln, zscore_opt=zscore_opt, skip_roi=skip_roi, variance_part=variance_part)
        
        # Save
        if save:
            folder_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/sub-{}{}/'.format(sub, suffix)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.savez('{}CCA_res_sub-{}_{}'.format(folder_path, sub, atlas_file), result_matrix=result_matrix, result_dict=result_dict, pca_dict=pca_dict)
        
        # Close textfile
        logfile.close()
    
    # Reset printing settings
    sys.stdout = orig_stdout

def pca_all_rois(sub, domains, threshold, critical_v, atlas=''):
    
    """
    Explore impact of PCA on all ROIs of a single subject

    Input:
    - sub : int, indicates subject number
    - domains : dict, including domain names as keys and domain regressors as values
    - threshold : str, indicates whether to specify n_components or min variance explained for PCA
    - critical_v : list, containing floats, inidicating thershold levels of variance explained to test
    - atlas : str, indicates atlas to use; default = '', which means atlas 200

    Output:
    - explained_variance : dict, containing ROIs as keys and array with cumulative sum of explained variance for each component as values
    - critical_c : dict, containing ROIs as keys and 
    - max_nc : int, 
    - n_rois_below : int,
    - rois_below : array, 
    - rois_to_exclude : array, 
    """

    # Load data
    data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-AV/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub)).get_fdata()
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/sub-{}_atlas{}_2orig.nii.gz'.format(sub, atlas)).get_fdata()
    labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N_labels.txt', dtype=str)

    # Extract rois
    data_rois, n_rois, n_voxels = extract_roi(data, atlas)
    minvoxs = np.sort(list(n_voxels.values()))
    maxpreds = np.max([v.shape[1] for v in domains.values()])
    minvox = minvoxs[np.argwhere(minvoxs >= maxpreds)[0][0]]

    print('### Subject {}'.format(sub))
    print('- n_rois: {}'.format(n_rois))
    print('- n_voxels between {} and {}'.format(np.min(list(n_voxels.values())), np.max(list(n_voxels.values()))))
    print('- n_components that would be used for PCA = {}'.format(minvox))

    # Initialize results dict
    explained_variance = {}
    
    if threshold == 'minvariance':
        # Perform PCA for each ROI
        for r, roi in data_rois.items():
            _, explained_variance[r] = pca_single_roi(roi, n_comps=None)

        # Test various critical values of explained variance
        for critical in critical_v:
            critical_c = {}
            
            for r in data_rois.keys():
                critical_c[r] = np.where(explained_variance[r] >= critical)[0][0]

            max_nc = np.max(np.array(list(critical_c.values())))
            n_rois_below = np.sum(list(n_voxels.values())<max_nc)
            rois_below = np.array(list(data_rois.keys()))[list(n_voxels.values())<max_nc]
            rois_to_exclude = labels[np.where(list(n_voxels.values())<max_nc)[0]]
            

            print('- n_components to reach at least {} of explained variance in all ROIs: {}'.format(critical, max_nc))
            print('- n_rois that would be excluded due to n_voxels below n_components: {}'.format(n_rois_below))
            print('ROIs to be escluded: {}'.format(rois_to_exclude))
        
        return explained_variance, critical_c, max_nc, n_rois_below, rois_below, rois_to_exclude

    elif threshold == 'components':
        
        # Perform PCA for each ROI
        for r, roi in data_rois.items():
            _, exvariance = pca_single_roi(roi, n_comps=minvox)
            explained_variance[r] = exvariance[-1]
        
        print('- explained variance within {} and {}'.format(np.min(list(explained_variance.values())), np.max(list(explained_variance.values()))))

        for critical in critical_v:
            n_rois_below = np.sum(np.array(list(explained_variance.values()))<critical)
            rois_below = np.array(list(data_rois.keys()))[np.array(list(explained_variance.values()))<critical]
            rois_to_exclude = labels[np.where(np.array(list(explained_variance.values()))<critical)[0]]
            print('- n_rois that would have explained variance lower than {}: {}'.format(critical, n_rois_below))
            print('ROIs with explained variance lower than {}: {}'.format(critical, rois_to_exclude))
        
        return explained_variance, n_rois_below, rois_below, rois_to_exclude
         
if __name__ == '__main__': 

    # Set parameters
    sub_list = np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32])
    atlas = ''

    # Print output to txt file
    log_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    logfile = open('{}log_pca_atlas{}.txt'.format(log_path, atlas), 'w')
    sys.stdout = logfile

    # Load task models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    n_doms = len(domains.keys())

    n_rois_below = np.zeros(len(sub_list))
    max_nc = np.zeros(len(sub_list))
    rois_to_exclude = {}
    rois_below = {}
    
    # Test PCA impact for all subs
    for s, sub in enumerate(sub_list): 
        explained_v, n_rois_below[s], rois_below[sub], rois_to_exclude[sub] = pca_all_rois(sub, domains, threshold='components', critical_v=[0.9], atlas=atlas)
        explained_v, critical_c, max_nc[s], n_rois_below[s], rois_below[sub], rois_to_exclude[sub] = pca_all_rois(sub, domains, threshold='minvariance', critical_v=[0.9], atlas=atlas)


    roistoex = []
    for x in list(rois_to_exclude.values()):
        for c in x:
            roistoex.append(c)

    rois_to_ex_dict = {x: roistoex.count(x) for x in np.unique(roistoex)}

    # Close textfile
    logfile.close()