import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

from sklearn.cross_decomposition import CCA
import numpy as np

from pareto_right_tail import pareto_right

def canoncorrelation(X,Y, center=True, adjust=True):
    
    """
    Performs canonical correlation analysis using sklearn.cross_decomposition CCA package
    
    Inputs:
    - X : matrix of shape n by d1
    - Y : matrix of shape n by d2
    - center: default = True; whether to remove the mean (columnwise) from each column of X and Y
    - adjust: default = True; whether to correct for number of predictor columns

    Outputs:
    - r2 : R-squared (Y*B = V), if adjusted = True, adjusted
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
    p = Y_pred.shape[1]
    if adjust:
        r2 = 1 - (1-r2)*((n-1)/(n-p-1))

    return r2, A, B, R, U, V

def permutation_schema(n_tpoints, n_perms=1000, chunk_size=15, seed=0, flip=True):
    
    """
    Creates an index permutation schema
    
    Inputs:
    - n_tpoints : number of timepoints (rows)
    - n_perms : number of permutations (columns); default = 1000
    - chunk_size: size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: seed for the random permutation; default = 0
    - flip: decide wether to flip or not random number of chunks, default = True

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

    # Flip some of the chunks
    if flip:
        flip_inds = np.random.choice([0, 1], size=len(chunks))
        chunks = [np.flip(chunk) if flip_inds[c] == 1 else chunk for c, chunk in enumerate(chunks)]

    # Shuffle the order of the chunks separately for each permutation
    for i in range(1,n_perms+1):
        perm_schema[:,i] = np.concatenate([list(chunks[c]) for c in np.random.permutation(len(chunks))])
        
    # Assign original indices to first column of permutation schema
    perm_schema[:,0] = indices

    return perm_schema

def run_canoncorr(roi, perm_schema, domains, adjust=True):
    
    """
    Run canonical correlation and store R2 between fMRI activity of a ROI and each domain model, first using real data and then permuted fMRI data
    
    Inputs:
    - roi : matrix containing fMRI data of shape = n_tpoints by n_voxels
    - perm_schema : matrix of shape = n_tpoints, n_perms; first row contains unshuffled indices --> contains indices for each permutation
    - domains : dictionary containing domains as keys and matrix of shape n_tpoints by n_columns as values
    - adjust : whether to get adjusted R2 or not, default = True
    
    Outputs:
    - results : matrix of shape = n_perms, n_domains; containing R2 values for each permutation and each domain
    """
    n_perms = perm_schema.shape[1]
    n_domains = len(domains)
    
    # Initialize results matrix
    results = np.full((n_perms, n_domains), np.nan)
    
    for perm in range(n_perms):
        order = perm_schema[:,perm] # shape = n_tpoints --> indices
        Y = roi[order,:] # reorder fmri signal based on permutation schema
        
        for d, domain in enumerate(domains.values()):
            X = domain # shape = n_tpoints by n_columns
            results[perm, d] = canoncorrelation(X, Y, adjust=adjust)[0]

    return results

def pvals(results, pareto=False):
    
    """
    Get p value for each domain
    
    Inputs:
    - results : matrix of shape = n_perms, n_domains; containing R2 values for each permutation and each domain
    - pareto : wether to do pareto; default = False

    Outputs:
    - pvalues : array of shape = n_domains
    """
    
    # Define number of domains and permutations
    n_domains = results.shape[1]
    n_perms = results.shape[0]-1
    
    # Get true R
    r_true = results[0,:]

    if pareto:
        # Pareto
        pvals = np.full((n_domains), np.nan)
        critical_values_at_p = np.full((n_domains), np.nan)
        for d in range(n_domains):
            pvals[d], critical_values_at_p[d] = pareto_right(results[1:,d], r_true[d], critical_p=0.05)

    else:
        # Sort permuted results
        res_sorted = np.sort(results, axis=0)

        # Get position of true result in sorted matrix for each domain
        positions = np.array([np.where(np.isclose(res_sorted[:,d], r_true[d]))[0][0] for d in range(n_domains)])

        # Calculate pval based on position
        pvals = 1-((positions)/(n_perms+1))

        # Get critical values at p

    return pvals, critical_values_at_p

def gen_correlated_data(realdata, n_voxels=100, noise=1):
    
    """
    Generates fake data starting from real data
    
    Inputs:
    - realdata : real data to be used as base for generated data
    - n_voxels : number of voxels of data to be generated; default = 100
    - noise : noise coefficient; default = 1

    Outputs:
    - fakedata : matrix of shape = n_tpoints, n_voxels
    """
    # Get number of timepoints and columns
    columns = realdata.shape[1]
    tpoints = realdata.shape[0]

    # Initialize matrix for fake data (shape = tpoints by n_voxels)
    fakedata = np.full((tpoints, n_voxels), np.nan)
    
    # Generate fake signal for each voxel
    for voxel in range(n_voxels):
        beta = np.random.randn(columns) # generate random beta coefficients, one for each column
        fakedata[:, voxel] = np.dot(realdata, beta) + noise*np.random.randn(tpoints) # multiply real data by betacoefficients and add noise (regulated by coefficient)
    
    return fakedata

def gen_fmri_signal(t_points=1614, tr=2, n_voxels=100):
    
    """
    Generates fake fmri data
    
    Inputs:
    - t_points : number of timepoints; default = 1614
    - tr : TR of fMRI signal (time resolution) in seconds; default = 2
    - n_voxels : number of voxels of data to be generated; default = 100

    Outputs:
    - fakedata : matrix of shape = n_tpoints, n_voxels
    """
    # Create an empty array to store the fMRI data for the set of voxels
    fakedata = np.random.rand(t_points, n_voxels)
    
    # Generate time array
    t = np.arange(t_points)

    # Simulate the fMRI signals for each voxel
    for voxel in range(n_voxels):
        phase_shift = np.random.randn() * np.pi * voxel / n_voxels
        fakedata[:, voxel] = np.sin(2 * np.pi * t / (tr * 30)+ phase_shift) + 0.5 * np.random.randn()
        
    return fakedata