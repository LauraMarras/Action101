import numpy as np
import os
from nilearn import image
from scipy.stats import zscore

from utils.exectime_decor import timeit
from simulation.semination_funcs import downsample_timeshift, seminate_mask
from simulation.noise_funcs import add_noise
from simulation.motion_funcs import add_motion
from simulation.trend_funcs import segment, add_trend

@timeit
def simulate_subject(sub, fmri_params, task_params, motion_params, seed_schema, options):

    """
    Simulate fMRI signal of a single subject

    Inputs:
    - sub : int, subject number to save data
    - fmri_params : dict, including:
        - TR : int or float, fMRI resolution, in seconds
        - R : float, desired maximum correlation coefficient between task signal and signal scaled by SNR with added noise
        - betas : int, float or array of shape = n_task_regressor(s), correlation coefficient(s) to scale task regressor(s), if None, they are drawn randomly
        - noise_level : int or float, indicates scale of gaussian noise
        - n_tissues : int, number of populations to segment the volume into (usually air/CSF/white/grey), for trend generation
        - n_bins_trend : int, number of bins for adding variation to trend coefficients within tissues
        - semination_mask : str, filename of nifti file containing semination mask
    - task_params : dict, including:
        - task_path : str, filepath of csv or txt file containing task regressors
        - task_time_res : float, sampling interval of task in seconds
        - n_runs : int, number of runs
        - run_cuts : array of shape = n_runs by 2, containing start and end timepoints of each run (end timepoint excluded)
    - motion_params : dict, including:
        - movement_upscale : int, factor to which upscale image if needed for motion, upscalefactor == 1 means no upscaling
        - regressors_path : str, path where real data movement regressors are stored
    - seed_schema : array of shape = n_subs by 14, containing seed numbers unique for each run and subject
    - options : dict, including:
        - add_noise_bool : bool, whether to add noise
        - add_trend_bool : bool, whether to add trend
        - add_motion_bool : bool, whether to add motion
        - save : bool, whether to save nifti files for each run

    Outputs:
    - saves nii images, one for each run: 4d matrix of shape x by y by z by time

    Calls:
    - downsample_timeshift()
    - add_noise()
    - seminate_mask()
    - segment()
    - add_trend()
    - add_motion()
    """

    # Load fMRI template and semination mask (voxels where to seminate task signal)
    template_nii = image.load_img('data/simulazione_datasets/sub-{}/func/template.nii.gz'.format(sub))
    template = template_nii.get_fdata()
    template_volume = template[:,:,:,0] # Get single volume
    semination_mask = image.load_img('data/simulazione_datasets/sub-{}/anat/{}'.format(sub, fmri_params['semination_mask'])).get_fdata()

    # Get n_voxels and slices, mean and std
    voxel_dims_mm = tuple(template_nii.header._structarr['pixdim'][1:4])
    x,y,slices,_ = template_nii.shape
    template_avg = np.mean(template, axis=3)
    template_std = np.std(template, axis=3, ddof=1)

    # Load task regressors
    task = np.loadtxt(task_params['task_path'], delimiter=',', skiprows=1)[:, 1:] # In this case we skip the first column as it contains indices
    task = np.atleast_2d(task.T).T # Make sure the matrix is 2d even when the regressor is only one

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice    
    task_downsampled_byslice = downsample_timeshift(task, slices, task_params['task_time_res'], fmri_params['TR'])

    # Initialize data matrix
    data_init = np.zeros((x, y, slices, task_downsampled_byslice.shape[1]))
    
    # Generate and add noise in all voxels   
    if options['add_noise_bool']:
        data_init = add_noise(data_init, fmri_params['noise_level'], fmri_params['TR'], seed_schema[-2], sub=sub, reference=template_nii)

    # Create fMRI signal starting from task and seminate only in mask
    data_signal = seminate_mask(task_downsampled_byslice, semination_mask, data_init, fmri_params['R'], fmri_params['betas'], seed=seed_schema[-1], sub=sub, reference=template_nii)

    # Segment
    tissues_mask = segment(template_avg, fmri_params['n_tissues'], use_threshold=False, plot=False, sub=sub, reference=template_nii)
    
    # Iterate over runs
    for r in range(task_params['n_runs']):
        print('# run {}'.format(r+1))

        # Get data of single run
        run_idx = [*range(task_params['run_cuts'][r][0], task_params['run_cuts'][r][1])]
        data_run = data_signal[:,:,:,run_idx]

        # Generate Trend (for each run separately)
        if options['add_trend_bool']:
            data_run = add_trend(data_run, template_volume, tissues_mask, n_bins=fmri_params['n_bins_trend'], seed=seed_schema[r], sub=sub, run=r+1, reference=template_nii)
           
        # Zscore
        run_zscore = zscore((data_run), axis=3, nan_policy='omit')
        data_run = run_zscore * np.expand_dims(template_std, axis=3) + np.expand_dims(template_avg, axis=3)

        # Add motion
        if options['add_motion_bool']:        
            data_run = add_motion(data_run, dimensions=voxel_dims_mm, upscalefactor=motion_params['movement_upscale'], regressors_path=motion_params['regressors_path'], seed=seed_schema[task_params['n_runs']+r], save=True, sub=sub, run=r+1) 
            
        # Save data
        if options['save']:
            folder_path = 'data/simulazione_results/sub-{}/fmri/'.format(sub)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            image_final = image.new_img_like(template_nii, data_run, affine=template_nii.affine, copy_header=True)
            image_final.header._structarr['slice_duration'] = fmri_params['TR']
            image_final.header._structarr['pixdim'][4] = fmri_params['TR']
            image_final.to_filename('{}simul_run{}.nii'.format(folder_path, r+1))