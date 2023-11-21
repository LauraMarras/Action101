import numpy as np
from nilearn import image
from scipy.stats import zscore as zscore
import time

def convolve_HRF(model_df, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12):
    
    """
    Convolve each column of matrix with a HRF 
    
    Inputs:
    - model_df : dataframe of full group model
    - tr : sampling interval in seconds (fMRI TR)
    - hrf_p : parameter of HRF
    - hrf_q : parameter of HRF
    - dur : duration of HRF, in seconds

    Outputs:
    - group_convolved : dataframe of full group model convolved
    """

    # Define HRF
    hrf_t = np.arange(0, dur+0.5, tr)  # A typical HRF lasts 12 secs
    hrf = (hrf_t / (hrf_p * hrf_q)) ** hrf_p * np.exp(hrf_p - hrf_t / hrf_q)

    # Initialize matrix to save result
    group_convolved = np.full(model_df.shape, np.nan)

    # Iterate over columns
    for column in range(model_df.shape[1]):
        model = model_df[:,column]
        model_conv = np.convolve(model, hrf, mode='full')[:model.shape[0]] # cut last part 
        model_conv = model_conv / np.max(model_conv)
        group_convolved[:,column] = model_conv
    
    return group_convolved




if __name__ == '__main__':
    t = time.time()
    # Define parameters
    n_points = 1614
    TR = 2
    time_res = 0.05
    SNR_base = 1
    noise_level = 4
    poly_deg = 4

    # Movement parameters
    movement_upscale = 6
    output_movement = 'datasets/fake_movement.csv'
    SNR_movement = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)

    # Define options
    add_noise = True
    add_trend = True
    add_motion = False

    # Load task data
    data_path = 'Data/Carica101_Models/Domains/tagging_carica101_group_2su3_'
    task = np.loadtxt(data_path + 'agent_objective.csv', delimiter=',', skiprows=1)[:, 1:]

    # Load fMRI data and Mask
    data = image.load_img('datasets/run1_template.nii')
    mask = image.load_img('datasets/atlas_2orig.nii')

    # Upsample task
    task_upsampled = np.zeros((int(n_points*TR/time_res), task.shape[1]))

    for t in range(n_points):
        start_index = round((t)*TR/time_res)
        end_index = round((t+1)*TR/time_res)
        task_upsampled[start_index:end_index,:] = np.tile(task[t,:], (end_index-start_index, 1))

    # Convolve
    task_convolved = convolve_HRF(task_upsampled, tr=TR, hrf_p=8.6, hrf_q=0.547, dur=12)

    # fMRI
    x,y,slices,_ = data.shape
    data_map = data.get_fdata()

    data_avg = np.mean(data_map, 3)
    data_std = np.std(data_map, 3, ddof=1)

    # Downsample convolved regressors back to TR resolution
    task_downsampled = np.zeros((n_points, task.shape[1]))
    task_downsampled_byslice = np.full((slices, n_points, task.shape[1]), np.nan)

    for t in range(n_points):
        task_downsampled[t,:] = task_convolved[int((t)*TR/time_res),:]
        for s in range(slices):
            task_downsampled_byslice[s,t,:] = task_convolved[int((t)*TR/time_res) + s,:]

    # Generate Noise and trend
    noise = np.random.randn(x, y, slices, n_points)*noise_level

    trend = np.zeros((x, y, slices, n_points))
    for i in range(x):
        for j in range(y):
            for s in range(slices):
                temp_s = zscore((data_map[i,j,s,:] - data_avg[i,j,s]))
                poly_coeffs = np.polyfit(np.arange(temp_s.shape[0]), temp_s, poly_deg)
                trend[i,j,s,:] = np.round(np.polyval(poly_coeffs, np.arange(n_points)))

    # Create fMRI signal
    mask_map = mask.get_fdata()

    data_signal = np.zeros((x, y, slices, n_points))
    betas = np.random.randn(task.shape[1])
    for s in range(slices):
        signal = np.dot(task_downsampled_byslice[s,:,:], betas)
        (x_inds, y_inds) = np.where(mask_map[:,:,s] == 1)
        data_signal[x_inds, y_inds, s, :] = signal*SNR_base

    # Add all:
    data_final = zscore((data_signal + noise + trend), 3)

    for t in range(n_points):
        data_final[:,:,:,t] = data_final[:,:,:,t]*data_std + data_avg

    # Add motion

    # Save data
    image_final = image.new_img_like(data, data_final, copy_header=True)
    image_final.to_filename('dati_simul.nii')

    print(time.time() - t)