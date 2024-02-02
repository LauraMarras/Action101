import numpy as np
import os
from nilearn import image
from utils.exectime_decor import timeit


@timeit
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

@timeit
def seminate_mask(task, ROI_mask, data_noise, r=0.5, betas=None, step=(0.01, 0.001), seed=0, sub=0, reference=None):
    
    """
    Create fMRI signal based task regressors and desired R correlation coefficient, and assign it to voxels within ROI

    Inputs:
    - task : array, 3d matrix of shape = n_timepoints (in TR resolution) by slices by n_task_regressor(s)
    - ROI_mask : array, 3d matrix of booleans of shape = x by y by z indicating voxel within ROI (where to seminate task signal)
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted random noise
    - r : float, desired maximum correlation coefficient between task signal and signal scaled by SNR with added noise; default = 0.5
    - betas : int, float or array of shape = n_task_regressor(s), correlation coefficient(s) to scale task regressor(s), if None, they are drawn randomly; default None
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

    # Set seed
    np.random.seed(seed)
    
    # Create signal by multiply task data by given or random betas
    if betas:
        betas = np.ones(task.shape[2]) * betas
    else:
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
    
    print('- semination parameters : SNR = {}; maximum R = {}'.format(SNR_corr, rmax))

    # Save signal, betas and data_noise
    if reference:
        folder_path = 'data/simulazione_results/sub-{}/semina/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt('{}signal.1D'.format(folder_path), signal, delimiter=' ')
        np.savetxt('{}betas.1D'.format(folder_path), betas, delimiter=' ')

        img = image.new_img_like(reference, data_noise, affine=reference.affine, copy_header=True)
        img.to_filename('{}noise_plus_signal.nii'.format(folder_path))

    return data_noise