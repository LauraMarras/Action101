import numpy as np
import os
from matplotlib import pyplot as plt
from nilearn import image
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from utils.exectime_decor import timeit

@timeit
def segment(volume, n_tissues=4, use_threshold=False, plot=None, sub=0, reference=None):
    
    """
    Segment volume into n populations of voxels based on intensity (tissues)

    Inputs:
    - volume : array, 3d matrix of shape x by y by z, original MRI volume
    - n_tissues : int, number of populations to be segmented; default = 4 (air, white matter, grey matter, csf)
    - use_threshold : bool, whether to not consider voxels supposedly located outside of brain based on threshold; default = False
    - plot : str, path where (whether) to save plotted histogram and gaussian of extracted populations; default = None
    - save : str, path where (whether) to save tissue mask as nifti file; default = None (don't save)
    - sub : int, subject number to save data; default = 0
    - reference : niilike object, used as reference to save data as nifti file; default = None
    
    Outputs:
    - mat_max : array, matrix of shape x by y by z indicating membership for each voxel
    - saves GMM.png figure
    """
    
    # Exclude voxels below threshold if given (supposedly voxels outside brain)
    if use_threshold:
        _, edges = np.histogram(volume.flatten(), 4, density=True)
        edges_center = edges[:-1] + np.diff(edges)/2
        threshold = edges_center[0]
        vflat = volume[np.where(volume>threshold)].flatten()
        aria_mask = volume <= threshold

    else:
        vflat = volume.flatten()
    
    # Obtain intensity distribution
    histogram, bin_edges = np.histogram(vflat, np.unique(vflat), density=True)
    bin_edges = bin_edges[:-1] + np.diff(bin_edges)/2
    
    # Run GaussianMixtureModel to discriminate each tissue gaussian curve
    gm = GaussianMixture(n_tissues, random_state=0)
    model = gm.fit(np.vstack((histogram, bin_edges)).T)
    idxs = np.argsort(model.means_[:,1])
    comp_means = model.means_[idxs, 1]
    comp_stds = np.sqrt(model.covariances_[idxs,1,1])

    # Initialize matrices
    mats = np.full((volume.shape[0], volume.shape[1], volume.shape[2], n_tissues), np.nan)

    # Assign each voxel tissue probability 
    for c in range(n_tissues):
        mats[:,:,:,c] = norm(comp_means[c], comp_stds[c]).pdf(volume)
    
    # Assign each voxel the tissue with highest probability
    mat_max = np.argmax(mats, axis=3)

    # Assign 0 to all voxels below threshold
    if use_threshold:
        mat_max[aria_mask] = 0

    # Plot
    if plot:
        distros = np.full((n_tissues, 10000), np.nan)
        x = np.full((n_tissues, 10000), np.nan)
        
        for c in range(n_tissues):
            gs = norm(comp_means[c], comp_stds[c])
            x[c,:] = np.sort(gs.rvs(size=10000))
            distros[c,:] = gs.pdf(x[c,:])

        plt.hist(volume.flatten(), np.unique(volume), density=True)
        plt.plot(x.T, distros.T)

        # Adjust
        plt.ylim(0, np.max(distros[1:]))

        # Save
        plt.savefig('{}GMM_{}{}.png'.format(plot, n_tissues, '_thresh' if use_threshold else ''))

    # Save
    if reference:
        folder_path = 'data/simulazione_results/sub-{}/segmentation/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img = image.new_img_like(reference, mat_max.astype('int32'), affine=reference.affine, copy_header=True)
        img.to_filename('{}segment_mask.nii'.format(folder_path))

    return mat_max

@timeit
def add_trend(data_run, volume, tissues_mask, scale_parameter=100, n_bins=1, seed=0, TR=2, sub=0, run=0, reference=None):
    
    """
    Generate trend signal for each voxel

    Inputs:
    - data_run : array, 4d matrix of shape x by y by z by time, fMRI signal of single run
    - volume : array, 3d matrix of shape x by y by z,  original MRI volume
    - tissues_mask : array, 3d matrix of shape x by y by z, indicating membership for each voxel (air, white matter, grey matter, csf)
    - scale_parameter : int, empirical estimated parameter to scale polynomial coefficients; default = 100
    - n_bins : int, number of bins for adding variation to trend coefficients within tissues; default = 1 (that is, no variation)
    - seed : int, seed for random generation of polynomial coefficients; default = 0
    - TR : int or float, fMRI resolution, in seconds; default = 2
    - sub : int, subject number to save data; default = 0
    - run :  int, run number to save data; default = 0
    - reference : niilike object, used as reference to save polynomial coefficients (4d matrix of shape x by y by z by poly_deg+1, indicating trend polynomial coefficients for each voxel) as nifti file; default = None (don't save)
    
    Outputs:
    - data_trend : array, 4d matrix of shape x by y by z by nTRs, containing fMRI signal with added trend timeseries for each voxel
    """

    # Get shape of run matrix
    x,y,z,nTRs = data_run.shape

    # Get number of tissues
    tissues = np.unique(tissues_mask)

    # Esitmate degree of the polynomial based on legth of each run
    poly_deg = 1 + round(TR*nTRs/150)

    # Initialize matrix of trend for each voxel get a time series of trends
    trend = np.zeros(data_run.shape)
    poly_coeff_mat = np.zeros((x,y,z, poly_deg+1))
    
    # Set seed
    np.random.seed(seed)

    # Generate random coefficients: matrix of poly coefficients (based on polydeg) for each tissue
    random_coeffs = np.random.randn(poly_deg+1, len(tissues))

    # Scale the random coeffients according to order and tissue mean intensity
    poly_sorted = np.sort(random_coeffs, axis=1)
    
    scale_poly_order = np.append(np.power(10, np.arange(2,(poly_deg)*2 +1,2))[::-1], 1)
    scale_tissue = np.array([np.mean(volume[np.where(tissues_mask == tissue)]) for tissue in tissues])/np.mean(volume)/scale_parameter
    
    poly_scaled = poly_sorted / scale_poly_order[:, None] * scale_tissue

    # Create trend time series for each tissue and assign to matrix using tissue mask
    for tissue in tissues[1:]:
        n_vox_tiss = np.sum(tissues_mask == tissue)
        bin_size = ((n_vox_tiss - n_vox_tiss%n_bins)/n_bins).astype(int)
        drawn = np.random.choice(n_vox_tiss-n_vox_tiss%n_bins, (n_bins, bin_size), replace=False)
        remaining = np.arange(n_vox_tiss)[~np.isin(np.arange(n_vox_tiss),drawn)]

        # Add jitter      
        for vox in range(n_bins):
            x_inds, y_inds, z_inds = (np.where(tissues_mask == tissue)[0][drawn[vox]], np.where(tissues_mask == tissue)[1][drawn[vox]], np.where(tissues_mask == tissue)[2][drawn[vox]])
            
            poly_jitt = np.abs(poly_scaled[:,tissue]/10*np.random.randn())+poly_scaled[:,tissue]
            trend[x_inds, y_inds, z_inds, :] = np.round(np.polyval(poly_jitt, np.arange(nTRs)))
            poly_coeff_mat[x_inds, y_inds, z_inds, :] = poly_jitt

        if len(remaining) != 0:
            x_inds, y_inds, z_inds = (np.where(tissues_mask == tissue)[0][remaining], np.where(tissues_mask == tissue)[1][remaining], np.where(tissues_mask == tissue)[2][remaining])
            poly_jitt = np.abs(poly_scaled[:,tissue]/10*np.random.randn())+poly_scaled[:,tissue]
            trend[x_inds, y_inds, z_inds, :] = np.round(np.polyval(poly_jitt, np.arange(nTRs)))
            poly_coeff_mat[x_inds, y_inds, z_inds, :] = poly_jitt
            
    # Add trend to data
    data_trend = data_run + trend

    # Save polynomial coefficients as nifti file
    if reference:
        folder_path = 'data/simulazione_results/sub-{}/trend/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img = image.new_img_like(reference, poly_coeff_mat, affine=reference.affine, copy_header=True)
        img.to_filename('{}polycoeffs_run{}.nii'.format(folder_path, run))

    return data_trend