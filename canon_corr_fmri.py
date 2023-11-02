import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from check_collinearity import canoncorrelation
import numpy as np

def permutation_schema(n_tpoints, n_perms=1000, chunk_size=15, seed=0):
    """
    Creates an index permutation schema
    
    Inputs:
    - n_tpoints : number of timepoints (rows)
    - n_perms : number of permutations (columns); default = 1000
    - chunk_size: size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: seed for the random permutation; default = 0

    Outputs:
    - perm_schema : matrix of shape = n_tpoints, n_perms + 1; first row contains unshuffled indices
    """
    # Set seed
    np.random.seed(seed)
    
    # Create an array of indices from 0 to number of timepoints
    indices = np.arange(n_tpoints)

    # Initialize the permutation schema matrix
    perm_schema = np.zeros((n_tpoints, n_perms+1), dtype=int)

    # Create chunks of contiguous timepoints
    chunks = [indices[i:i + chunk_size] for i in range(0, n_tpoints, chunk_size)]

    # Shuffle the order of the chunks separately for each permutation
    for i in range(1,n_perms+1):
        perm_schema[:,i] = np.concatenate([list(chunks[c]) for c in np.random.permutation(len(chunks))])
        
    # Assign original indices to first column of permutation schema
    perm_schema[:,0] = indices

    return perm_schema

def run_canoncorr(roi, data, perm_schema, domains):
    """
    Run canonical correlation and store R2 between fMRI activity of a ROI and each domain model, first using real data and then permuted fMRI data
    
    Inputs:
    - roi : number of ROI (index of roi in data matrix) # to be changed to name of dictionary key?
    - data : matrix containing fMRI data of shape = n_tpoints by n_voxels by n_rois # to be changed to dictionary containing rois as keys and matrix of shape n_tpoints by n_voxels as values
    - perm_schema : matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - domains : dictionary containing domains as keys and matrix of shape n_tpoints by n_columns as values

    Outputs:
    - results : matrix of shape = n_perms, n_domains; containing R2 values for each permutation and each domain
    """
    n_perms = perm_schema.shape[1]
    n_domains = len(domains)
    
    data_roi = data[roi] # get fmri data of specific roi --> shape = n_tpoints by n_voxels

    # Initialize results matrix
    results = np.full((n_perms, n_domains), np.nan)
    
    for perm in range(n_perms):
        order = perm_schema[:,perm] # shape = n_tpoints --> indices
        Y = data_roi[order,:] # reorder fmri signal based on permutation schema
        
        for d, domain in enumerate(domains.values()):
            X = domain # shape = n_tpoints by n_columns
            results[perm, d] = canoncorrelation(X, Y)[0]

    return results

def generate_signal(t_points=1614, tr=2, n_voxels=100):

    # Create an empty array to store the fMRI data for the set of voxels
    fmri_data = np.random.rand(t_points, n_voxels)

    # Simulate the fMRI signals for each voxel
    for voxel in range(n_voxels):
        phase_shift = np.random.randn() * np.pi * voxel / n_voxels
        for t in range(1, t_points):
            # For a simple example, let's use a sine wave for each voxel.
            fmri_data[t, voxel] = np.sin(2 * np.pi * t / (tr * 30)+ phase_shift) + 0.5 * np.random.randn()

    return fmri_data