import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from sklearn.cross_decomposition import CCA
import numpy as np
import multiprocessing as mp
from utils.exectime_decor import timeit
from nilearn import image
import sys
from datetime import datetime

from permutation_schema_func import permutation_schema

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

def run_cca_single_roi(roi, perm_schema, domains):
    
    """
    Run canonical correlation and store R2 between fMRI activity of a ROI and each domain model, first using real data and then permuted fMRI data
    
    Inputs:
    - roi : array, 2d matrix of shape = n_tpoints by n_voxels containing fMRI data 
    - perm_schema : array, 2d matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - domains : dict, including domains as keys and 2d matrix of shape = n_tpoints by n_columns as values
    
    Outputs:
    - results : array, 3d matrix of shape = 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each permutation and each domain
        
    Calls:
    - canonical_correlation()
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
        for d, domain in enumerate(domains.values()):
            X = domain
            
            r2, r2adj, _, _, _, _, _ = canonical_correlation(X, Y)
            results[0, perm, d] = r2
            results[1, perm, d] = r2adj

    return results

def extract_roi(data, atlas):
    
    """
    Parcellize data into ROIs of given atlas
    
    Inputs:
    - data : array, 4d (or 3d) matrix of shape = x by y by z (by time) containing fMRI signal
    - atlas : array, 3d matrix of shape = x by y by z containing ROIs membership for each voxel

    Outputs:
    - data_rois : dict, including for each ROI within atlas, 2d (or 1d) matrix of shape = voxels (by time) containing fMRI signal of each voxel within ROI
    - n_rois : int, number of ROIs in atlas
    - n_voxels_rois : dict, including for each ROI within atlas, the number of voxels within ROI
    """
    
    # Get list and number of ROIs in atlas
    rois = np.unique(atlas)
    rois = np.delete(rois, np.argwhere(rois==0))
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
def run_cca_all_rois(data_rois, domains, perm_schema, pooln=20):
    
    """
    Run canonical correlation for all ROIs using parallelization
    
    Inputs:
    - data_rois : dict, including for each ROI within atlas, 2d (or 1d) matrix of shape = voxels (by time) containing fMRI signal of each voxel within ROI
    - domains : dict, including domains as keys and 2d matrix of shape = n_tpoints by n_columns as values
    - perm_schema : array, 2d matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - pooln : int, number of parallelization processes; default = 20
    
    Outputs:
    - result_matrix : array, 4d matrix of shape = n_rois by 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each ROI, each permutation and each domain
    - result_dict : dict, including ROI as keys and result matrix of each ROI as values
    
    Calls:
    - run_cca_single_roi()
    """

    # Define number of ROIs and of permutations
    n_rois = len(data_rois.keys())
    n_perms = perm_schema.shape[1]
    
    # Initialize results matrix and dictionary
    result_matrix = np.empty((n_rois, 2, n_perms, len(domains)))
    result_dict = {}

    # Set pool
    results_pool = []
    pool = mp.Pool(pooln)

    # Run canoncorr for each ROI with parallelization and store results
    for r, roi in data_rois.items():
        result_pool = pool.apply_async(run_cca_single_roi, args=(roi, perm_schema, domains))
        results_pool.append((r, result_pool))
    pool.close()
    
    # Unpack results
    for result_pool in results_pool:
        njob = result_pool[1]._job
        roi_n = result_pool[0]
        result_matrix[njob, :, :, :] = result_pool[1].get()
        result_dict[roi_n] = result_pool[1].get()

    return result_matrix, result_dict

def run_cca_all_subjects(sub_list, atlas_file, n_perms=1000, chunk_size=15, seed=0, pooln=20, save=True):
    
    """
    Run canonical correlation for all subjects
    
    Inputs:
    - sub_list : array, 1d array containing sub numbers for which to run CCA
    - atlas_file : str, filename of atlas to use
    - n_perms : int, number of permutations (columns); default = 1000
    - chunk_size: int, size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: int, seed for the random permutation; default = 0
    - pooln : int, number of parallelization processes; default = 20
    - save : bool, whether to save results as npy files; default = True
    
    Saves:
    For each subject
    - result_matrix : array, 4d matrix of shape = n_rois by 2 by n_perms by n_domains; containing R2 values (not adjusted and adjusted) for each ROI, each permutation and each domain
    - result_dict : dict, including ROI as keys and result matrix of each ROI as values
    
    Calls:
    - extract_roi()
    - permutation_schema()
    - run_cca_all_rois()
    """

    # Load task models
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}

    # Set printing settings
    orig_stdout = sys.stdout
    
    # Iterate over subjects
    for sub in sub_list:
        
        # Print output to txt file
        log_path = 'data/cca_results/sub-{}/logs/'.format(sub)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        logfile = open('{}log_cca_sub-{}_{}.txt'.format(log_path, sub, 'Schaefer200' if atlas_file == 'atlas_2orig' else 'Schaefer1000'), 'w')
        sys.stdout = logfile

        print(datetime.now())
        print('CCA of sub-{}'.format(sub))
        print('- n_perms: {}'.format(n_perms))
        print('- seed: {}'.format(seed))
        print('\n- Atlas: {}'.format('Schaefer200' if atlas_file == 'atlas_2orig' else 'Schaefer1000'))

        # Load data
        data = image.load_img('data/simulazione_preprocessed/sub-{}/func/cleaned.nii.gz'.format(sub)).get_fdata()
        atlas = image.load_img('data/simulazione_datasets/sub-{}/anat/{}.nii.gz'.format(sub, atlas_file)).get_fdata()
        
        # Extract rois
        data_rois, n_rois, n_voxels = extract_roi(data, atlas)
        
        print('- n_rois: {}'.format(n_rois))
        print('- n_voxels between {} and {}'.format(np.min(list(n_voxels.values())), np.max(list(n_voxels.values()))))

        # Generate permutation schema
        n_tpoints = data.shape[-1]
        perm_schema = permutation_schema(n_tpoints, n_perms=n_perms, chunk_size=chunk_size)

        # Run cca for each roi
        result_matrix, result_dict = run_cca_all_rois(data_rois, domains, perm_schema, pooln=pooln)
        
        # Save
        if save:
            folder_path = 'data/cca_results/sub-{}/'.format(sub)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.savez('{}CCA_res_sub-{}_{}'.format(folder_path, sub, 'Schaefer200' if atlas_file == 'atlas_2orig' else 'Schaefer1000'), result_matrix=result_matrix, result_dict=result_dict)
        
        # Close textfile
        logfile.close()
    
    # Reset printing settings
    sys.stdout = orig_stdout