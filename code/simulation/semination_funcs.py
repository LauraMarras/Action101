import numpy as np
import os
from nilearn import image
from utils.exectime_decor import timeit


@timeit
def downsample_timeshift(task, n_slices, TR=2):
    
    """
    Downsample task data to TR resolution and create time shift over slices

    Inputs:
    - task : array, 2d matrix of shape n_timepoints (in task time resolution) by n_task_regressor(s)
    - n_slices : int, number of fMRI volume slices
    - TR : int or float, fMRI resolution, in seconds; default = 2
   
    Outputs:
    - task_downsampled_byslice : array, 2d matrix of shape n_timepoints (in TR resolution) by n_task_regressor(s)
    """

    # Get dimensions
    task_time_res = np.round((TR/n_slices), 2)
    n_points = int(task.shape[0]/TR*task_time_res)

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice
    task_downsampled_byslice = np.full((n_slices, n_points, task.shape[1]), np.nan)
    for t in range(n_points):
        for s in range(n_slices):
            task_downsampled_byslice[s,t,:] = task[int(t*TR/task_time_res) + s]
    
    return task_downsampled_byslice

def create_signal(task, data_noise, x_inds, y_inds, z_inds, betas=None, n_bins=0, seed=0):
    
    """
    Create fMRI signal based on task regressors and assign it to voxels within ROI

    Inputs:
    - task : array, 3d matrix of shape = n_timepoints (in TR resolution) by slices by n_task_regressor(s)
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted random noise
    - x_inds : array, 1d array containing x coordinates of voxels within ROI mask
    - y_inds : array, 1d array containing y coordinates of voxels within ROI mask
    - z_inds : array, 1d array containing z coordinates of voxels within ROI mask
    - betas : int, float or array of shape = n_task_regressor(s), correlation coefficient(s) to scale task regressor(s), if None, they are drawn randomly; default None
    - n_bins : int, number of bins for adding variation to betas across voxels; default = 0 (that is, no variation)
    - seed : int, seed for random generation of betas; default = 0
  
    Outputs:
    - signal : array, 4d matrix of shape = x by y by slices containing signal in voxels within ROI and nothing elsewhere
    - betas : array, 2d (or 1d) matrix of shape = n_bins (or n_bins +1) by n_task_regressor(s) containing correlation coefficient(s) to scale task regressor(s)
    """

    # Initialize signal matrix
    signal = np.empty(data_noise.shape)

    # Set seed
    np.random.seed(seed)
    
    # If betas are not given generate random betas
    if betas:
        betas = np.ones(task.shape[2]) * betas
    else:
        betas = np.abs(np.random.randn(task.shape[2]))
    
    # Create signal by multiplying task data by betas
    signal[x_inds, y_inds] = np.dot(task, betas)

    # If requested (n_bins not 0), add jitter to betas (slightly different beta across voxels)
    if n_bins > 0:
        
        # Geberate randomly jittered betas within given jitter range (+-1/10 of beta value in this case)
        jitter = betas/10
        betas_jittered = np.random.uniform(betas-jitter, betas+jitter, (n_bins, betas.shape[0]))    
        
        # Get number of voxels within semination mask and accordingly bin size
        n_voxels = len(x_inds)
        bin_size = int(np.ceil((n_voxels-n_voxels%n_bins)/n_bins))
        
        # Divide mask into bins (spatially close voxels will get same beta)
        x_bins = np.reshape(x_inds[:(n_voxels - (n_voxels%n_bins))], (n_bins, bin_size))
        y_bins = np.reshape(y_inds[:(n_voxels - (n_voxels%n_bins))], (n_bins, bin_size))
        z_bins = np.reshape(z_inds[:(n_voxels - (n_voxels%n_bins))], (n_bins, bin_size))
        
        # Create signal with jittered betas and assign to voxels within each bin     
        for bin in range(n_bins):
            signal[x_bins[bin], y_bins[bin], z_bins[bin]] = np.dot(task, betas_jittered[bin])[z_bins[bin]]

        # Assign signal with original beta to "spare" voxels within mask (voxels that did not get into any bin)
        if n_voxels%n_bins > 0:
            x_rem = x_inds[-(n_voxels%n_bins):]
            y_rem = y_inds[-(n_voxels%n_bins):]
            z_rem = z_inds[-(n_voxels%n_bins):]
            
            signal[x_rem, y_rem, z_rem] = np.dot(task, betas)[z_rem]
            betas_jittered = np.concatenate((np.expand_dims(betas, 0), betas_jittered))
    
        betas = betas_jittered
        
    return signal, betas

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
    signal_noise = data_noise[x_inds, y_inds, z_inds, :] + (signal_scale * signal[x_inds, y_inds, z_inds].T).T

    # Iterate over voxels within ROI_mask
    for vox in range(len(x_inds)):
        signal_vox = signal[x_inds[vox], y_inds[vox], z_inds[vox]]
        signal_noise_vox = signal_noise[vox]

        # Compute correlation between scaled and noisy signal and original clean signal
        r = np.corrcoef(signal_vox, signal_noise_vox)[0][1:]
        r = 0 if np.isnan(r) else r
        rlist = np.append(rlist, r)
    
    # Compute R max across slices
    rmax = np.max(rlist)

    return rmax, SNR, signal_noise

@timeit
def seminate_mask(task, ROI_mask, data_noise, r=0.5, betas=None, n_bins=0, step=(0.01, 0.001), seed=0, sub=0, reference=None):
    
    """
    Create fMRI signal based task regressors and desired R correlation coefficient, and assign it to voxels within ROI

    Inputs:
    - task : array, 3d matrix of shape = n_timepoints (in TR resolution) by slices by n_task_regressor(s)
    - ROI_mask : array, 3d matrix of booleans of shape = x by y by z indicating voxel within ROI (where to seminate task signal)
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted random noise
    - r : float, desired maximum correlation coefficient between task signal and signal scaled by SNR with added noise; default = 0.5
    - betas : int, float or array of shape = n_task_regressor(s), correlation coefficient(s) to scale task regressor(s), if None, they are drawn randomly; default None
    - n_bins : int, number of bins for adding variation to betas across voxels; default = 0 (that is, no variation) 
    - step : tuple, tuple of len = 2, indicating gridsearch intervals; default = (0.01, 0.001)
    - seed : int, seed for random generation of betas; default = 0
    - sub : int, subject number to save data; default = 0
    - reference : niilike object, used as reference to save data as nifti file, if None don't save; default = None
    
    Outputs:
    - data_noise : array, 4d matrix of shape = x by y by z by time containing task-related signal only in voxels within mask
    
    Prints:
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
    
    # Create signal
    signal, betas = create_signal(task, data_noise, x_inds, y_inds, z_inds, betas, n_bins, seed=seed)

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
    
    print('- semination parameters : SNR = {}; maximum R = {}'.format(SNR_corr, rmax))

    # Save signal, betas and data_noise
    if reference:
        folder_path = 'data/simulazione_results/sub-{}/semina/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt('{}betas.1D'.format(folder_path), betas, delimiter=' ')

        img = image.new_img_like(reference, data_noise, affine=reference.affine, copy_header=True)
        img.to_filename('{}noise_plus_signal.nii'.format(folder_path))

        img = image.new_img_like(reference, signal, affine=reference.affine, copy_header=True)
        img.to_filename('{}signal.nii'.format(folder_path))

    return data_noise