import numpy as np
import time
import sys
import os
from matplotlib import pyplot as plt
from nilearn import image
from scipy.stats import zscore, norm
from scipy.ndimage import zoom, affine_transform
from skimage import transform
from sklearn.mixture import GaussianMixture

def downsample_timeshift(task, n_slices, task_time_res=0.05, TR=2):
    
    """
    Downsample task data to TR resolution and create time shift over slices

    Inputs:
    - task : array, 2d matrix of shape n_timepoints (in task time resolution) by n_task_regressor(s)
    - n_slices : int, number of fMRI volume slices
    - task_time_res : float, sampling interval of task in seconds; default = 0.05
    - TR : int or float, fMRI resolution, in seconds; default = 2
   
    Outputs:
    - task_downsampled_byslice : array, 2d matrix of shape n_timepoints (in TR resolution) by n_task_regressor(s)
    """

    # Get dimensions
    n_points = int(task.shape[0]/TR*task_time_res)

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice
    task_downsampled_byslice = np.full((n_slices, n_points, task.shape[1]), np.nan)
    for t in range(n_points):
        for s in range(n_slices):
            task_downsampled_byslice[s,t,:] = task[int(t*TR/task_time_res) + s]
    
    return task_downsampled_byslice

def compute_correlation(SNR, data_noise, signal, x_inds, y_inds, z_inds, seed=0):
    
    """
    Compute correlation coefficient between task signal and signal scaled by SNR with added noise

    Inputs:
    - SNR : float, signal scaler
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted rnadom noise
    - signal : array, 2d matrix of shape = slices by timepoints
    - x_inds : array, 1d array containing x coordinates of voxels within ROI mask
    - y_inds : array, 1d array containing y coordinates of voxels within ROI mask
    - z_inds : array, 1d array containing z coordinates of voxels within ROI mask
    - seed : int, seed for random generation of betas; default = 0

    Outputs:
    - rmax : float, maximum correlation coefficient between task signal and signal scaled by SNR with added noise, across voxels within ROI mask
    - SNR : float, SNR value used to scale the signal
    - signal_noise : array, 4d matrix of shape = x by y by z by time containing task-related signal + noise in voxels within mask, else noise
    """

    # Set seed
    np.random.seed(seed+1)
    
    # Initialize array of Rs (one for each slice)
    rlist = np.array([])
    
    # Create SNR array (pick random one for each voxel within slice)
    signal_scale = (np.random.randn(x_inds.shape[0]) * SNR/10) + SNR                    
    
    # Scale signal by SNR and assign signal to each voxel within mask (add to noise)
    signal_noise = data_noise[x_inds, y_inds, z_inds, :] + (signal[z_inds].T*signal_scale).T

    # Iterate over slices within ROI_mask
    for slice in np.unique(z_inds):

        # Compute correlation between scaled and noisy signal and original clean signal
        r = np.corrcoef(signal[slice], signal_noise[np.where(z_inds == slice)])[0][1:]
        rlist = np.append(rlist, r)

    # Compute R max across slices
    rmax = np.max(rlist)

    return rmax, SNR, signal_noise

def seminate_squarewave(ROI_mask, data_noise, r=0.3, step=(0.01, 0.001), seed=0, save=None):
    
    """
    Create fMRI signal based task regressors and desired R correlation coefficient, and assign it to voxels within ROI

    Inputs:
    - ROI_mask : array, 3d matrix of booleans of shape = x by y by z indicating voxel within ROI
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted random noise
    - r : float, desired maximum correlation coefficient between task signal and signal scaled by SNR with added noise; default = 0.3
    - step : tuple, tuple of len = 2, indicating gridsearch intervals; default = (0.01, 0.001)
    - seed : int, seed for random generation of betas; default = 0
    - save : str, path where (whether) to save signal and beta coefficients as 1D files; default = None
    
    Outputs:
    - signal_noise : array, 4d matrix of shape = x by y by z by time containing task-related signal only in voxels within mask
    - SNR_corr : float, final SNR value used to scale the signal
    - rmax : float, final correlation coefficient obtained

    Calls:
    - compute_correlation()
    """
    
    # Get mask indices
    (x_inds, y_inds, z_inds) = np.where(ROI_mask == 1)
    
    SNR = 0
    rmax = 0
    step_1 = step[0]
    step_2 = step[1]

    # Set seed
    np.random.seed(seed)
    
    # Create signal
    signal = np.ones((data_noise.shape[2], data_noise.shape[-1]))
    signal[:, 400:500] = 2

    # Verify that desired R falls within reasonable range     
    if r < 0:
        print('The desired R is < 0, considered |R|')
    
    if r >= 1:
        print('The desired R is >= 1, considered R = 0.9999')
        r = np.sign(r) * 0.9999

    # Start from SNR = 0, increase SNR by step until the obtained R is larger or equal to |desired R|
    while rmax < np.abs(r):
        rmax, SNR_corr, signal_noise = compute_correlation(SNR, data_noise, signal, x_inds, y_inds, z_inds, seed=seed+1)
        SNR += step_1

    # Verify that the |desired R| isn't lower than the one obtained by chance (correlating task signal with random noise)
    if SNR_corr == 0:
        print('The |desired R| is lower than the R obtained by correlating task signal with random noise, considered SNR = 0')           

    else:
        # Starting from last value of SNR, decrease SNR by step until the obtained R is lower or equal to |desired R|
        while rmax > np.abs(r):
            
            # Save results from previous iteration
            signal_noise_prev = signal_noise
            SNR_prev = SNR_corr
            rmax_prev = rmax
            
            SNR = SNR_corr - step_2
            
            # Check that you don't use negative SNR (in case you are really close to SNR = 0), in case, go to previous value of SNR (positive), reduce your step by 1/10
            if SNR <= 0:
                SNR += step_2
                step_2 = step_2/10
                SNR -= step_2

            # Stop when your step is too small (10^-5), you are either really close to SNR==0 or to |desired R|
            if step_2 <= 10**(-5):
                break
            
            rmax, SNR_corr, signal_noise = compute_correlation(SNR, data_noise, signal, x_inds, y_inds, z_inds, seed=seed+1)
    
        if np.abs(rmax - np.abs(r)) > np.abs(rmax_prev - np.abs(r)):
            signal_noise = signal_noise_prev
            SNR_corr = SNR_prev
            rmax = rmax_prev

    data_noise[x_inds, y_inds, z_inds, :] = signal_noise
    
    # Save signal and data
    if save:
        np.savetxt('data/simulazione_results/{}square_signal.1D'.format(save), signal, delimiter=' ')
        save_images(data_noise, '{}data_noise'.format(save))
    
    return data_noise, SNR_corr, rmax

def seminate_mask(task, ROI_mask, data_noise, r=0.3, step=(0.01, 0.001), seed=0, save=None):
    
    """
    Create fMRI signal based task regressors and desired R correlation coefficient, and assign it to voxels within ROI

    Inputs:
    - task : array, 3d matrix of shape = n_timepoints (in TR resolution) by slices by n_task_regressor(s)
    - ROI_mask : array, 3d matrix of booleans of shape = x by y by z indicating voxel within ROI
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted rnadom noise
    - r : float, desired maximum correlation coefficient between task signal and signal scaled by SNR with added noise; default = 0.3
    - step : tuple, tuple of len = 2, indicating gridsearch intervals; default = (0.01, 0.001)
    - seed : int, seed for random generation of betas; default = 0
    - save : str, path where (whether) to save signal and beta coefficients as 1D files; default = None
    
    Outputs:
    - signal_noise : array, 4d matrix of shape = x by y by z by time containing task-related signal only in voxels within mask
    - SNR_corr : float, final SNR value used to scale the signal
    - rmax : float, final correlation coefficient obtained

    Calls:
    - compute_correlation()
    """
    
    # Get mask indices
    (x_inds, y_inds, z_inds) = np.where(ROI_mask == 1)
    
    SNR = 0
    rmax = 0
    step_1 = step[0]
    step_2 = step[1]

    # Set seed
    np.random.seed(seed)
    
    # Create signal by multiply task data by random betas
    betas = np.abs(np.random.randn(task.shape[2]))
    signal = np.dot(task, betas)

    # Verify that desired R falls within reasonable range     
    if r < 0:
        print('The desired R is < 0, considered |R|')
    
    if r >= 1:
        print('The desired R is >= 1, considered R = 0.9999')
        r = np.sign(r) * 0.9999

    # Start from SNR = 0, increase SNR by step until the obtained R is larger or equal to |desired R|
    while rmax < np.abs(r):
        rmax, SNR_corr, signal_noise = compute_correlation(SNR, data_noise, signal, x_inds, y_inds, z_inds, seed=seed+1)
        SNR += step_1

    # Verify that the |desired R| isn't lower than the one obtained by chance (correlating task signal with random noise)
    if SNR_corr == 0:
        print('The |desired R| is lower than the R obtained by correlating task signal with random noise, considered SNR = 0')           

    else:
        # Starting from last value of SNR, decrease SNR by step until the obtained R is lower or equal to |desired R|
        while rmax > np.abs(r):
            
            # Save results from previous iteration
            signal_noise_prev = signal_noise
            SNR_prev = SNR_corr
            rmax_prev = rmax
            
            SNR = SNR_corr - step_2
            
            # Check that you don't use negative SNR (in case you are really close to SNR = 0), in case, go to previous value of SNR (positive), reduce your step by 1/10
            if SNR <= 0:
                SNR += step_2
                step_2 = step_2/10
                SNR -= step_2

            # Stop when your step is too small (10^-5), you are either really close to SNR==0 or to |desired R|
            if step_2 <= 10**(-5):
                break
            
            rmax, SNR_corr, signal_noise = compute_correlation(SNR, data_noise, signal, x_inds, y_inds, z_inds, seed=seed+1)
    
        if np.abs(rmax - np.abs(r)) > np.abs(rmax_prev - np.abs(r)):
            signal_noise = signal_noise_prev
            SNR_corr = SNR_prev
            rmax = rmax_prev

    data_noise[x_inds, y_inds, z_inds, :] = signal_noise

    # Save signal and betas
    if save:
        np.savetxt('data/simulazione_results/{}signal.1D'.format(save), signal, delimiter=' ')
        np.savetxt('data/simulazione_results/{}betas.1D'.format(save), betas, delimiter=' ')
        save_images(data_noise, '{}data_noise'.format(save))

    return data_noise, SNR_corr, rmax

def add_noise(data_signal, noise_level=4, TR=2, seed=0, save=None, check_autocorr=False):
    
    """
    Add noise to fMRI data matrix

    Inputs:
    - data_signal : array, 4d matrix of shape = x by y by z by time containing task-related signal only in voxels within mask, else zeros or containing only zeros
    - noise_level : int or float, indicates scale of gaussian noise; default = 4
    - TR : int or float, fMRI resolution, in seconds; default = 2
    - seed : int, seed for random generation of gaussian noise; default = 0
    - save : str, path where (whether) to save generated noise as nifti file; default = None (don't save)
    - check_autocorr : bool, whether to measure difference in autocorrelation pre and post convolution or not; default = False
    
    Outputs:
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted noise and task-related signal
    
    Calls:
    - convolve_HRF()
    - save_images()
    - autocorr_diff()
    """

    # Get fMRI data dimensions
    (x, y, z, n_points) = data_signal.shape
    
    # Set seed
    np.random.seed(seed)

    # Create gaussian noise
    noise = np.random.randn(x, y, z, n_points)*noise_level

    # Convolve noise with HRF
    noise_conv = convolve_HRF(noise, TR)

    # Add noise to signal
    data_noise = data_signal + noise_conv

    # Save noise
    if save:
        save_images(noise_conv, save)
    
    # Check difference in autocorrelation
    if check_autocorr:
        autocorr_diff(data_signal+noise, data_noise, coords=(64,94,17)) #(56,92,16)

    return data_noise

def convolve_HRF(mat, TR=2, hrf_p=8.6, hrf_q=0.547, dur=12):
    
    """
    Convolve each column of matrix with a HRF 
    
    Inputs:
    - mat : array, 4d matrix of shape x by y by z by time to be convolved
    - TR : int or float, fMRI resolution, in seconds; default = 2
    - hrf_p : int or float, parameter of HRF; default = 8.6
    - hrf_q : int or float, parameter of HRF; default = 0.547
    - dur : int or float, duration of HRF in seconds; default = 12

    Outputs:
    - mat_conv : array, 4d matrix of shape x by y by z by time
    """

    # Define HRF
    hrf_t = np.arange(0, dur+0.005, TR)  # A typical HRF lasts 12 secs
    hrf = (hrf_t / (hrf_p * hrf_q)) ** hrf_p * np.exp(hrf_p - hrf_t / hrf_q)

    # Initialize matrix to save result
    mat_conv = np.full(mat.shape, np.nan)

    # Iterate over dimensions
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                mat_conv[x,y,z,:] = np.convolve(mat[x,y,z,:], hrf, mode='full')[:mat.shape[3]] # cut last part
    
    return mat_conv

def autocorr_diff(data_noise, data_noise_conv, coords=(0,0,0)):
    
    """
    Measure and plot autocorrelation of two voxels 
    
    Inputs:
    - data_noise : array, 4d matrix of shape x by y by z by time
    - data_noise_conv : array, 4d matrix of shape x by y by z by time
    - coords : tuple, sequence of 3 int indicating x, y and z coordinates of chosen voxel; default = (0,0,0)

    Outputs:
    - saves autocorrelation.png figure
    """

    x,y,z = coords
    # Plot the mean autocorrelation of random voxel
    fig, axs = plt.subplots(1,2, sharey='all')
    axs[0].acorr(data_noise[x,y,z,:], maxlags=30)
    axs[1].acorr(data_noise_conv[x,y,z,:], maxlags=30)

    # Set titles, labels
    axs[0].set_title('PRE convolution')
    axs[1].set_title('POST convolution')
    
    for ax in fig.get_axes():
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
    
    plt.suptitle('Autocorrelation of random voxel')
    plt.savefig('trash/autocorrelation_{}.png'.format(str(coords)))

def segment(volume, n_tissues=4, use_threshold=False, plot=False, save=None):
    
    """
    Segment volume into n populations of voxels based on intensity (tissues)

    Inputs:
    - volume : array, 3d matrix of shape x by y by z, original MRI volume
    - n_tissues : int, number of populations to be segmented; default = 4 (air, white matter, grey matter, csf)
    - use_threshold : bool, whether to not consider voxels supposedly located outside of brain based on threshold; default = False
    - plot : bool, whether to plot histogram and gaussian of extracted populations; default = False
    - save : str, path where (whether) to save tissue mask as nifti file; default = None (don't save)
    
    Outputs:
    - mat_max : array, matrix of shape x by y by z indicating membership for each voxel

    Calls:
    - save_images()
    """
    
    pname = 'GMM_{}'.format(n_tissues)

    if use_threshold: # size = (127,2,37)
        _, edges = np.histogram(volume.flatten(), 4, density=True)
        edges_center = edges[:-1] + np.diff(edges)/2
        threshold = edges_center[0]
        vflat = volume[np.where(volume>threshold)].flatten()
        aria_mask = volume <= threshold
        pname+='_thresh'

    else:
        vflat = volume.flatten()
    
    histogram, bin_edges = np.histogram(vflat, np.unique(vflat), density=True)
    bin_edges = bin_edges[:-1] + np.diff(bin_edges)/2
    
    gm = GaussianMixture(n_tissues, random_state=0)
    model = gm.fit(np.vstack((histogram, bin_edges)).T)
    idxs = np.argsort(model.means_[:,1])
    comp_means = model.means_[idxs, 1]
    comp_stds = np.sqrt(model.covariances_[idxs,1,1])

    mats = np.full((volume.shape[0], volume.shape[1], volume.shape[2], n_tissues), np.nan)
    for c in range(n_tissues):
        mats[:,:,:,c] = norm(comp_means[c], comp_stds[c]).pdf(volume)
        
    mat_max = np.argmax(mats, axis=3)

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

        hist = plt.hist(volume.flatten(), np.unique(volume), density=True)
        plt.plot(x.T, distros.T)

        # Adjust
        plt.ylim(0, np.max(distros[1:]))

        plt.savefig('{}.png'.format(pname))

    # Save
    if save:
        save_images(mat_max, save)

    return mat_max

def add_trend(data_run, volume, tissues_mask, scale_parameter=100, n_bins=1, seed=0, TR=2, save=None):
    
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
    - save : str, path where (whether) to save polynomial coefficients (4d matrix of shape x by y by z by poly_deg+1, indicating trend polynomial coefficients for each voxel) as nifti file; default = None (don't save)
    
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
        resto = np.arange(n_vox_tiss)[~np.isin(np.arange(n_vox_tiss),drawn)]
             
        for v in range(n_bins):
            xinds, y_inds, z_inds = (np.where(tissues_mask == tissue)[0][drawn[v]], np.where(tissues_mask == tissue)[1][drawn[v]], np.where(tissues_mask == tissue)[2][drawn[v]])
            
            poly_jitt = np.abs(poly_scaled[:,tissue]/10*np.random.randn())+poly_scaled[:,tissue]
            trend[xinds, y_inds, z_inds, :] = np.round(np.polyval(poly_jitt, np.arange(nTRs)))
            poly_coeff_mat[xinds, y_inds, z_inds, :] = poly_jitt

        if len(resto) != 0:
            xinds, y_inds, z_inds = (np.where(tissues_mask == tissue)[0][resto], np.where(tissues_mask == tissue)[1][resto], np.where(tissues_mask == tissue)[2][resto])
            poly_jitt = np.abs(poly_scaled[:,tissue]/10*np.random.randn())+poly_scaled[:,tissue]
            trend[xinds, y_inds, z_inds, :] = np.round(np.polyval(poly_jitt, np.arange(nTRs)))
            poly_coeff_mat[xinds, y_inds, z_inds, :] = poly_jitt
            
    # Add trend to data
    data_trend = data_run + trend

    # Save polynomial coefficients as nifti file
    if save:
        save_images(poly_coeff_mat, save)

    return data_trend

def add_motion(data_run, dimensions, regressors_path, upscalefactor=1, seed=0, save=None):
    
    """
    Generate motion regressors and transform signal for each voxel

    Inputs:
    - data :  array, 4d matrix of shape x by y by z by time, fMRI signal of single run
    - dimensions : tuple, dimension of voxels in mm, used to scale offsets; default = (2,2,3)
    - regressors_path : str, path where real data movement regressors are stored
    - upscalefactor : int, factor to which upscale image if needed, upscalefactor == 1 means no upscaling; default = 1
    - seed : int, seed for random selection of motion regressors real data; default = 0
    - save : str, path where (whether) to save motion regressors as 1D file; default = None (don't save)
    
    Outputs:
    - run_motion : array, 4d matrix of shape x by y by z by nTRs, containing fMRI signal with added motion for each voxel
    
    Calls:
    - get_movoffset_fromdata()
    - affine_transformation()
    """
    
    # Get movement offsets
    movement_offsets, mov_offs_notscaled = get_movoffset_fromdata(data_run.shape[-1], regressors_path, dimensions=dimensions, seed=seed)
    
    # Initialize final matrix
    run_motion = np.full(data_run.shape, np.nan)
    
    # Apply affine transform to each timepoint
    for t in range(data_run.shape[-1]):
        run_motion[:,:,:, t] = affine_transformation(data_run[:,:,:,t], movement_offsets[t,:], upscalefactor=upscalefactor)
    
    # Save movemet offsets --> save non scaled
    if save:
        np.savetxt('data/simulazione_results/{}.1D'.format(save), mov_offs_notscaled, delimiter=' ')
            
    return run_motion

def get_movoffset_fromdata(nTRs, regressors_path, dimensions=(2,2,3), seed=0):
    
    """
    Generate movement offsets signal along time starting from real data

    Inputs:
    - nTRs : int, number of TRs of wanted signal
    - regressors_path : str, path where real data movement regressors are stored
    - dimensions : tuple, dimension of voxels in mm, used to scale offsets; default = (2,2,3)
    - seed : int, seed for random selection of motion regressors real data; default = 0

    Outputs:
    - offset_signals_scaled : array, 2d matrix of shape nTRs by n_dims*2, signal for each movement offset, scaled by voxel dimensions
    - offset_signals : array, 2d matrix of shape nTRs by n_dims*2, signal for each movement offset, not scaled
    """

    # Load movement regressors of real subjects
    sublist = os.listdir(regressors_path)

    # Set seed
    np.random.seed(seed)

    # Initialize offset array
    offset_signals = np.full((nTRs, len(dimensions)*2), np.nan)

    # 
    temp = nTRs
    c = 0
    while temp > 0:
        sub = np.genfromtxt(regressors_path + sublist.pop(np.random.randint(0, len(sublist))) + '/derivatives/rest_mocopar.1D')
        idx = np.min((temp, sub.shape[0]))

        sub = sub - sub[0,:]         

        if c > 0:
            lastrow = offset_signals[c-1,:]
            offset_signals[c:idx+c, :] = sub[:idx,:] +lastrow

        else:
            offset_signals[c:idx+c, :] = sub[:idx,:]

        c+=len(sub)
        temp -= len(sub)
    
    # scale
    offset_signals_scaled = offset_signals / np.array([1,1,1, dimensions[0], dimensions[1], dimensions[2]])
    return offset_signals_scaled, offset_signals

def affine_transformation(volume, movement_offsets, upscalefactor=1, printtimes=False):
    
    """
    Applies affine transform to MRI volume given rotation and traslation offsets

    Inputs:
    - volume : array, 3d matrix of shape x by y by z, MRI volume at specific timepoint
    - movement_offsets : array, 1d array of shape 6 (3 rotation and 3 traslation)
    - upscalefactor : int, factor to which upscale image, upscalefactor == 1 means no upscaling; default = 1
    - printtimes : bool, whether to print times for each operation (upscaling, transform, downscaling); default = False
    
    Outputs:
    - trans_volume : array, d matrix of shape x by y by z, transformed MRI volume
    """

    tstart = time.time()
    
    # Upsample volume
    if upscalefactor != 1:
        volume = zoom(volume, upscalefactor, mode='nearest', order=0)
    tupscale = time.time() - tstart

    # Create rotation, shift and translation matrices
    angles = -np.radians(np.array([movement_offsets[1], movement_offsets[2], movement_offsets[0]]))
    shift = -np.array(volume.shape)/2 # shift to move origin to the center
    displacement = -np.array([movement_offsets[4], movement_offsets[5], movement_offsets[3]])
    
    r = transform.SimilarityTransform(rotation=angles, dimensionality=3)
    s = transform.SimilarityTransform(translation=shift, dimensionality=3)
    t = transform.SimilarityTransform(translation=displacement, dimensionality=3)
    
    # Compose transforms by multiplying their matrices (mind the order of the operations)
    trans_matrix = t.params @ np.linalg.inv(s.params) @ r.params @ s.params
   
    # Apply affine transform
    trans_volume = affine_transform(volume, np.linalg.inv(trans_matrix))
    ttransform = time.time() - tstart - tupscale

    # Scale down to original resolution
    if upscalefactor != 1:
        trans_volume = zoom(trans_volume, 1/upscalefactor, mode='nearest', order=0)
    tdownscale = time.time() - tstart - ttransform

    if printtimes:
        print('Time to upscale:{}s \nTime to transform:{}s \nTime to downscale:{}s'.format(tupscale, ttransform, tdownscale))
    
    return trans_volume

def save_images(img_tosave, path):
    
    """
    Save matrix as nifti file 
    
    Inputs:
    - img_tosave : array, 4d or 3d matrix to save as image
    - path : str, subfolders and filename

    !!! USES GLOBAL VARIABLE data_nii !!!
    """

    img = image.new_img_like(data_nii, img_tosave, affine=data_nii.affine, copy_header=True)
    img.to_filename('data/simulazione_results/{}.nii'.format(path))

def simulation_pipeline(n_subs, add_noise_bool, add_trend_bool, add_motion_bool, save, filename_suffix='debug'):
    
    # Profiling
    tstart = time.time()

    # Save options
    filename_prefix = 'simul'

    # Print output to txt file
    orig_stdout = sys.stdout
    logfile = open('data/simulazione_results/logs/out{}.txt'.format(filename_suffix), 'w')
    sys.stdout = logfile

    # Define fMRI parameters
    TR = 2
    np.random.seed(0)
    R = np.random.uniform(0, 1, n_subs)
    noise_level = 4
    n_tissues = 4 # air, white matter, grey matter, csf
    n_bins_trend = 80

    # Movement parameters
    movement_upscale = 1
    regressors_path = 'data/simulazione_datasets/motionreg/'
    
    # Load task data
    data_path = 'data/models/domains/group_us_conv_'
    task = np.loadtxt(data_path + 'agent_objective.csv', delimiter=',', skiprows=1)[:, 1:]
    task = np.atleast_2d(task.T).T

    # Define task parameters
    task_time_res = 0.05
    run_dur_sec = [536,450,640,650,472,480] # duration of each run in seconds
    run_dur_TR = (np.array(run_dur_sec)/TR).astype(int) # duration of each run in TRs
    run_cuts = np.array(list(zip((np.cumsum(run_dur_TR) - run_dur_TR), np.cumsum(run_dur_TR))))
    n_runs = len(run_dur_sec)

    # Create seed schema
    seed_schema = np.reshape(np.arange(0,((n_runs*2)+2)*n_subs), (n_subs,n_runs*2+2))

    # Sub loop
    for sub in range(n_subs):

        # Load fMRI data and Mask (voxels where to seminate task signal)
        global data_nii
        data_nii = image.load_img('data/simulazione_datasets/sub-0{}/run1_template.nii'.format(sub+1))
        mask_nii = image.load_img('data/simulazione_datasets/sub-0{}/mask_2orig.nii.gz'.format(sub+1))

        fmri_data = data_nii.get_fdata()[:,:,:,0] # Get single volume
        semination_mask = mask_nii.get_fdata()

        # Get n_voxels and slices, mean and std
        voxel_dims_mm = tuple(data_nii.header._structarr['pixdim'][1:4])
        x,y,slices,_ = data_nii.shape

        data_avg = np.mean(data_nii.get_fdata()[:,:,:,:10], axis=3) # estrarre 10 tpoints
        data_std = np.std(data_nii.get_fdata()[:,:,:,:10], axis=3, ddof=1)

        print('Sub {}. Done with: loading data, defining parameters. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Downsample convolved regressors back to TR resolution and add timeshift for each slice    
        task_downsampled_byslice = downsample_timeshift(task, slices, task_time_res, TR)
        print('Sub {}. Done with: downsampling and adding timeshift. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Generate and add noise in all voxels
        data_init = np.zeros((x, y, slices, task_downsampled_byslice.shape[1]))       
        if add_noise_bool:
            filename_prefix += '_noise'
            data_init = add_noise(data_init, noise_level, TR, seed_schema[sub,-2], save='sub{}/noise/noise_{}'.format(sub+1, filename_suffix), check_autocorr=False) #save=filename_suffix
        print('Sub {}. Done with: generating random noise. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Create fMRI signal starting from task and seminate only in mask
        data_signal, SNR, rmax = seminate_squarewave(semination_mask, data_init, R[sub], seed=seed_schema[sub,-1], save='sub{}/semina/{}'.format(sub+1, filename_suffix))
        #data_signal, SNR, rmax = seminate_mask(task_downsampled_byslice, semination_mask, data_init, R[sub], seed=seed_schema[sub,-1],save='sub{}/semina/{}'.format(sub+1, filename_suffix))
        print('Sub {}. Creating fMRI signal from task. Used SNR = {} to reach maximum R = {}'.format(sub+1, SNR, rmax))
        print('Sub {}. Done with: creating fMRI signal from task. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Segment
        tissues_mask = segment(data_avg, n_tissues, use_threshold=False, plot=False, save='sub{}/mask/mask_{}'.format(sub+1, filename_suffix)) #save=None
        print('Sub {}. Done with: segmenting. It took:  {}  seconds'.format(sub+1, time.time() - tstart))
        
        # Iterate over runs
        for r in range(n_runs):

            # Set filename for each run
            filename_prefixr = ''
            filename_suffixr = '_run{}_{}'.format(r+1, filename_suffix)

            # Get data of single run
            run_idx = [*range(run_cuts[r][0], run_cuts[r][1])]
            data_run = data_signal[:,:,:,run_idx]

            # Generate Trend (for each run separately)
            if add_trend_bool:
                filename_prefixr += '_trend' 
                data_run = add_trend(data_run, fmri_data, tissues_mask, n_bins=n_bins_trend, seed=seed_schema[sub,r], save='sub{}/trend/polycoeffs{}'.format(sub+1, filename_suffixr)) # save=None
            print('Sub {}. Done with: generating trend for run {}. It took:  {}  seconds'.format(sub+1, r+1, time.time() - tstart))
                
            # Zscore
            run_zscore = zscore((data_run), axis=3, nan_policy='omit')
            data_run = run_zscore * np.expand_dims(data_std, axis=3) + np.expand_dims(data_avg, axis=3) #problema se carichiamo solo un volume
            print('Sub {}. Done with: zscoring for run {}. It took:  {}  seconds'.format(sub+1, r+1, time.time() - tstart))

            # Add motion
            if add_motion_bool:        
                filename_prefixr += '_motion'
                data_run = add_motion(data_run, dimensions=voxel_dims_mm, upscalefactor=movement_upscale, regressors_path=regressors_path, seed=seed_schema[sub, n_runs+r], save='sub{}/motionreg/movement_offs{}'.format(sub+1, filename_suffixr)) # save=None
            print('Sub {}. Done with: adding motion for run {}. It took:  {}  seconds'.format(sub+1, r+1, time.time() - tstart))
                
            # Save data
            if save: 
                image_final = image.new_img_like(data_nii, data_run, affine = data_nii.affine, copy_header=True)
                image_final.header._structarr['slice_duration'] = TR
                image_final.header._structarr['pixdim'][4] = TR
                image_final.to_filename('data/simulazione_results/sub{}/fmri/{}{}.nii'.format(sub+1,filename_prefix+filename_prefixr, filename_suffixr))
            
        print('Sub {}. Done with: all. It took:  {}  seconds'.format(sub+1, time.time() - tstart))
    
    
    print('Done with: all {} subjects. It took:  {}  seconds'.format(n_subs, time.time() - tstart))
    
    # Close logfile
    sys.stdout = orig_stdout
    logfile.close()

    print('finished')

if __name__ == '__main__':
    
    # Define options
    n_subs = 1
    add_noise_bool = True
    add_trend_bool = True
    add_motion_bool = True
    
    # Saving options
    save = True
    filename_suffix = 'square_sig'
    
    # Call Pipeline
    simulation_pipeline(n_subs, add_noise_bool, add_trend_bool, add_motion_bool, save, filename_suffix)