import numpy as np
from nilearn import image
from scipy.stats import zscore as zscore
import time

from motion_regressors import generate_movement_regressors, rotate_mri

if __name__ == '__main__':
    tstart = time.time()
    # Define parameters
    n_points = 1614
    TR = 2
    time_res = 0.05
    SNR_base = 5
    noise_level = 4
    run_cuts = (np.array([536,450,640,650,472,480])/TR).astype('int')
    poly_deg = 1 + round(TR*(run_cuts.sum()/len(run_cuts))/150)

    # Movement parameters
    movement_upscale = 6
    output_movement = 'datasets/fake_movement.csv'
    SNR_movement = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)

    # Define options
    add_noise = True
    add_trend = True
    add_motion = True
    save = True

    # Load task data
    data_path = 'Data/Carica101_Models/Domains/group_us_conv_'
    task = np.loadtxt(data_path + 'agent_objective.csv', delimiter=',', skiprows=1)[:, 1:]

    # Load fMRI data and Mask
    data = image.load_img('datasets/run1_template.nii')
    mask = image.load_img('datasets/atlas_2orig.nii')
    data_map = data.get_fdata()
    mask_map = mask.get_fdata()

    # Get n_voxels and slices, mean and std
    x,y,slices,_ = data.shape
    data_avg = np.mean(data_map, 3)
    data_std = np.std(data_map, 3, ddof=1)

    print('Done with: loading data, defining parameters. It took:    ', time.time() - tstart, '  seconds')

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice
    task_downsampled_byslice = np.full((slices, n_points, task.shape[1]), np.nan)

    for t in range(n_points):
        for s in range(slices):
            task_downsampled_byslice[s,t,:] = task[int(t*TR/time_res) + s,:]

    print('Done with: downsampling and adding timeshift. It took:    ', time.time() - tstart, '  seconds')

    # Create fMRI signal starting from task
    data_signal = np.zeros((x, y, slices, n_points))
    betas = np.random.randn(task.shape[1])
    for s in range(15, slices):
        signal = np.dot(task_downsampled_byslice[s,:,:], betas)
        (x_inds, y_inds) = np.where(mask_map[:,:,s] == 1)
        noise = np.repeat(np.random.randn(x, y, 1)+SNR_base, signal.shape[0], axis=2)
        data_signal[x_inds, y_inds, s, :] = signal*noise[x_inds, y_inds, :]


    print('Done with: creating fMRI signal from task. It took:    ', time.time() - tstart, '  seconds')

    # Generate Noise
    if add_noise:
        noise = np.random.randn(x, y, slices, n_points)*noise_level
        data_signal += noise
    
    print('Done with: generating and adding noise. It took:    ', time.time() - tstart, '  seconds')

    
    idx=0
    for r, run_len in enumerate(run_cuts[:2]):
        run_idx = [*range(idx, run_len+idx)]
        
        # Get data of single run 
        data_run = data_signal[:,:,:,run_idx]

        # Generate Trend (for each run separately)
        if add_trend:
            trend = np.zeros((x, y, slices, run_len))
            for i in range(x):
                for j in range(y):
                    for s in range(slices):
                        temp_s = zscore((data_map[i,j,s,:] - data_avg[i,j,s]))
                        poly_coeffs = np.polyfit(np.arange(temp_s.shape[0]), temp_s, poly_deg)
                        trend[i,j,s,:] = np.round(np.polyval(poly_coeffs, np.arange(run_len)))

            data_run += trend
            print('Done with: generating trend for run {}. It took:    '.format(r), time.time() - tstart, '  seconds')
            
    
        # Zscore
        run_zscore = np.full(data_run.shape, np.nan)
        run_zscore = zscore((data_run), 3)        
        for t in range(run_len):
            run_zscore[:,:,:,t] = run_zscore[:,:,:,t]*data_std + data_avg
        
        print('Done with: zscoring for run {}. It took:    '.format(r), time.time() - tstart, '  seconds')
    
        # Add motion
        if add_motion:
            movement_offsets = generate_movement_regressors(run_len, SNR_movement)
            run_motion = np.full(run_zscore.shape, np.nan)
            for t in range(run_len):
                run_motion[:,:,:, t] = rotate_mri(run_zscore[:,:,:,t], movement_upscale, movement_offsets[t,:])
            print('Done with: adding motion for run {}. It took:    '.format(r), time.time() - tstart, '  seconds')
        
        # Save data
            if save:
                image_final = image.new_img_like(data, run_motion, copy_header=True)
                image_final.to_filename('run_motion_{}.nii'.format(r))

        else:
            if save:
                image_final = image.new_img_like(data, run_zscore, copy_header=True)
                image_final.to_filename('run_final_{}.nii'.format(r))
        
        idx+=run_len

    print('Done with: all. It took:    '.format(r), time.time() - tstart, '  seconds')
        