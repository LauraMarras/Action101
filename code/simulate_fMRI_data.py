import numpy as np
from nilearn import image
from scipy.stats import zscore
from scipy.ndimage import zoom, affine_transform
from skimage import transform
import pandas as pd
import time
from matplotlib import pyplot as plt
import sys

from motion_regressors import get_motion_offsets_data

def convolve_HRF(fMRI, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12):
    
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
    hrf_t = np.arange(0, dur+0.005, tr)  # A typical HRF lasts 12 secs
    hrf = (hrf_t / (hrf_p * hrf_q)) ** hrf_p * np.exp(hrf_p - hrf_t / hrf_q)

    # Initialize matrix to save result
    fMRIconv = np.full(fMRI.shape, np.nan)

    # Iterate over dimensions
    for x in range(fMRI.shape[0]):
        for y in range(fMRI.shape[1]):
            for z in range(fMRI.shape[2]):
                fMRIconv[x,y,z,:] = np.convolve(fMRI[x,y,z,:], hrf, mode='full')[:fMRI.shape[3]] # cut last part
    
    return fMRIconv

def get_movement_offsets(nTRs, SNR, dims=3, window_size=3, seed=0):
    
    """
    Generate movement offsets signal along time

    Inputs:
    - nTRs : int, number of TRs of wanted signal
    - SNR : tuple of int, signal to noise, determines how much to scale offsets for rotation and traslation
    - dims : int, dimensionality of volume; default = 3
    - window_size : int, size (number of TRs) of window used for smoothing signal; default = 3
    - seed : seed for the random generation; default = 0

    Outputs:
    - offset_signals : signal for each movement offset, matrix of shape nTRs by dims*2
    """

    # Set seed
    np.random.seed(seed)

    # Set scaling
    scaling = np.concatenate(((np.random.randn(dims)/SNR[0]), (np.random.randn(dims)/SNR[1])), 0)
        
    # Create single signal
    x = np.arange(0,nTRs+window_size)
    deg = np.random.randint(2,7)
    poly_coeffs = np.random.randn(deg)
    signal_fit = np.polyval(poly_coeffs,x)
    signal_fit = zscore(signal_fit) + np.random.randn(len(x))
    signal_fit = signal_fit/np.std(signal_fit)
    signal_series = pd.Series(signal_fit)
    signal_fit = signal_series.rolling(window_size+1).mean()

    # Scale signal for each scaling and create offset signals
    offsets_signals = np.full((nTRs, dims*2), np.nan)
    for p in range(dims*2):
        trend_scaled = signal_fit*scaling[p]
        offsets_signals[:,p] = trend_scaled[window_size:(nTRs+window_size)]
    
    return offsets_signals

def affine_transformation(volume, movement_offsets, upscalefactor=6, printtimes=False):
    
    """
    Applies affine transform to MRI volume given rotation and traslation offsets

    Inputs:
    - volume : original MRI volume, matrix of shape x by y by z
    - movement_offsets : movement offsets, array of shape 6 (3 rotation and 3 traslation)
    - upscalefactor : factor to which upscale image, int, upscalefactor == 1 means no upscaling; default = 6
    - printtimes : bool, whether to print times for each operation (upscaling, transform, downscaling); default = False
    
    Outputs:
    - trans_volume : transformed volume, matrix of shape x by y by z
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

def plot_transform(original, transformed, off, xyz=(64, 64, 19), save=None, cross=True):
    
    """
    Plots 3d view of original and transformed MRI volumes

    Inputs:
    - original : matrix of shape x by y by s
    - transformed : matrix of shape x by y by s (output of affine_transform)
    - off : movement offsets, array of shape 6 (3 rotation and 3 traslation)
    - xyz : tuple of len=3 indicating slices to show; default = (64, 64, 19)
    - save : filename to save figure, if want to save; default = None
    - cross : whether to add crosses indicating slices; default = True
    
    Outputs:
    - saves or shows figure
    """
    
    # Get slice coords
    x,y,s = xyz
    
    # Create figure with 6 subplots
    fig, axs = plt.subplots(3,2, gridspec_kw=dict(height_ratios=[128/38, 1, 1], width_ratios=[1,1]),  sharex=True, sharey=False)
    
    # Axial
    axs[0,0].imshow(original[:,:,s])
    axs[0,1].imshow(transformed[:,:,s])
   
    # Sagittal
    axs[1,0].imshow(original[x,:,:].T)
    axs[1,1].imshow(transformed[x,:,:].T)
    
    # Coronal
    axs[2,0].imshow(original[:,y,:].T)
    axs[2,1].imshow(transformed[:,y,:].T)
    
    # Invert axes
    for ax in fig.axes:
        ax.invert_yaxis()
    
    # Add crosses
    if cross:
        axs[0,0].axvline(y, lw=0.5, ls='--', color='r')
        axs[0,1].axvline(y, lw=0.5, ls='--', color='r')
        axs[0,0].axhline(x, lw=0.5, ls='--', color='r')
        axs[0,1].axhline(x, lw=0.5, ls='--', color='r')

        axs[1,0].axvline(y, lw=0.5, ls='--', color='r')
        axs[1,1].axvline(y, lw=0.5, ls='--', color='r')
        axs[1,0].axhline(s, lw=0.5, ls='--', color='r')
        axs[1,1].axhline(s, lw=0.5, ls='--', color='r')

        axs[2,0].axvline(x, lw=0.5, ls='--', color='r')
        axs[2,1].axvline(x, lw=0.5, ls='--', color='r')
        axs[2,0].axhline(s, lw=0.5, ls='--', color='r')
        axs[2,1].axhline(s, lw=0.5, ls='--', color='r')

    # Add Titles
    axs[0,0].set_title('Original Volume', fontsize=12)
    axs[0,1].set_title('Transformed Volume', fontsize=12)

    axs[0,0].set_ylabel('Axial \n(z={})'.format(s))
    axs[1,0].set_ylabel('Sagittal \n(x={})'.format(x))
    axs[2,0].set_ylabel('Coronal \n(y={})'.format(y))

    # Add transformation parameters
    off = np.round(off,3)
    plt.text(0.01, 0.99,
             'traslation:\nx={}  y={}  z={}'.format(off[3], off[4], off[5]),
             verticalalignment='top', horizontalalignment='left',
             transform=plt.gcf().transFigure,
             color='purple', fontsize=9)
    
    plt.text(0.51, 0.99,
             'rotation:\npitch={}  roll={}  yaw={}'.format(off[0], off[1], off[2]),
             verticalalignment='top', horizontalalignment='left',
             transform=plt.gcf().transFigure,
             color='green', fontsize=9)

    # Save
    if save:
        plt.savefig(save)
    else:
        plt.show()


if __name__ == '__main__':
    print('starting')

    # Define options
    add_noise = True
    add_trend = True
    add_motion = True
    save_motion = False
    save_polycoeff = False
    save = True
    trialn= '_8'

    orig_stdout = sys.stdout
    f = open('data/simulazione_results/logs/out{}.txt'.format(trialn), 'w')
    sys.stdout = f

    tstart = time.time()
    
    # Define parameters
    n_points = 1614
    TR = 2
    time_res = 0.05
    SNR_base = 5
    noise_level = 4
    run_cuts = (np.array([536,450,640,650,472,480])/TR).astype('int')
    poly_deg = 1 + round(TR*(run_cuts.sum()/len(run_cuts))/150)
    
    fname='simul'

    # Movement parameters
    movement_upscale = 1
    regressors_path = 'data/simulazione_datasets/motionreg/'
    
    # Set seed
    n_runs = len(run_cuts)
    n_subs = 10
    seed_mat = np.reshape(np.arange(0,(n_runs+2)*n_subs), (n_subs,n_runs+2)) 
    ### da mettere prima di for loop per soggetti

    # Sub loop
    sub = 0 

    # Load task data
    data_path = 'data/models/Domains/group_us_conv_'
    task = np.loadtxt(data_path + 'agent_objective.csv', delimiter=',', skiprows=1)[:, 1]

    # Load fMRI data and Mask
    data = image.load_img('data/simulazione_datasets/run1_template.nii')
    mask = image.load_img('data/simulazione_datasets/atlas_2orig.nii')
    data_map = data.get_fdata()
    mask_map = mask.get_fdata()

    # Get n_voxels and slices, mean and std
    dimensions = tuple(data.header._structarr['pixdim'][1:4])
    x,y,slices,_ = data.shape
    data_avg = np.mean(data_map, 3)
    data_std = np.std(data_map, 3, ddof=1)

    print('Done with: loading data, defining parameters. It took:    ', time.time() - tstart, '  seconds')

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice
    if len(task.shape) > 1:
        task_columns = task.shape[1]
    else:
        task_columns = 1
    
    task_downsampled_byslice = np.full((slices, n_points, task_columns), np.nan)
    for t in range(n_points):
        for s in range(slices):
            task_downsampled_byslice[s,t,:] = task[int(t*TR/time_res) + s]

    print('Done with: downsampling and adding timeshift. It took:    ', time.time() - tstart, '  seconds')

    # Create fMRI signal starting from task
    data_signal = np.zeros((x, y, slices, n_points))
    betas = np.random.randn(task_columns)
    
    # Set seed
    np.random.seed(seed_mat[sub,-1])
    
    for s in range(slices):
        signal = np.dot(task_downsampled_byslice[s,:,:], betas)
        (x_inds, y_inds) = np.where(mask_map[:,:,s] == 1)
        noise = np.repeat(np.random.randn(x, y, 1)+SNR_base, signal.shape[0], axis=2)
        data_signal[x_inds, y_inds, s, :] = signal*noise[x_inds, y_inds, :]

    print('Done with: creating fMRI signal from task. It took:    ', time.time() - tstart, '  seconds')

    # Generate Noise
    if add_noise:
        fname+='_noise'
        
        # Set seed
        np.random.seed(seed_mat[sub,-2])

        # Create gaussian noise
        noise = np.random.randn(x, y, slices, n_points)*noise_level
    
        # Convolve noise with HRF
        noise_conv = convolve_HRF(noise, tr=TR)

        # Add to data
        data_signal += noise_conv

    print('Done with: generating and adding noise. It took:    ', time.time() - tstart, '  seconds')

    
    idx=0
    for r, run_len in enumerate(run_cuts):
        fnamer = ''

        # Set seed 
        np.random.seed(seed_mat[sub,r])

        run_idx = [*range(idx, run_len+idx)]
        
        # Get data of single run 
        data_run = data_signal[:,:,:,run_idx]

        # Generate Trend (for each run separately)
        if add_trend:
            fnamer+='_trend'
            trend = np.zeros((x, y, slices, run_len))
            poly_coeffs_mat = np.zeros((x, y, slices, poly_deg+1))

            for i in range(x):
                for j in range(y):
                    for s in range(slices):
                        temp_s = zscore((data_map[i,j,s,:] - data_avg[i,j,s]))
                        if not np.any(np.isnan(temp_s)):
                            poly_coeffs_mat[i,j,s,:] = np.polyfit(np.arange(temp_s.shape[0]), temp_s, poly_deg)
                            trend[i,j,s,:] = np.round(np.polyval(poly_coeffs_mat[i,j,s,:], np.arange(run_len)))
                            
            # Salvare arr coefficienti su nifti
            if save_polycoeff:
                poly_coeffs_img = image.new_img_like(data, poly_coeffs_mat, copy_header=True)
                poly_coeffs_img.to_filename('data/simulazione_results/trend/polycoeffs_run{}_{}.nii'.format(r+1, trialn))

            data_run += trend
            print('Done with: generating trend for run {}. It took:    '.format(r+1), time.time() - tstart, '  seconds')
            
    
        # Zscore
        run_zscore = zscore((data_run), 3, nan_policy='omit')        
        for t in range(run_len):
            run_zscore[:,:,:,t] = run_zscore[:,:,:,t]*data_std + data_avg
        
        print('Done with: zscoring for run {}. It took:    '.format(r+1), time.time() - tstart, '  seconds')

        
        # Add motion
        if add_motion:
            fnamer+='_motion'
            movement_offsets = get_motion_offsets_data(run_len, regressors_path, dimensions=dimensions)
            run_motion = np.full(run_zscore.shape, np.nan)

            for t in range(run_len):
                run_motion[:,:,:, t] = affine_transformation(run_zscore[:,:,:,t], movement_offsets[t,:], upscalefactor=movement_upscale, printtimes=False)
            
            print('Done with: adding motion for run {}. It took:    '.format(r+1), time.time() - tstart, '  seconds')
        
            
        # Save data
            if save:
                fnamer+='_run{}'.format(r+1)
                image_final = image.new_img_like(data, run_motion, affine = data.affine, copy_header=True)
                image_final.header._structarr['slice_duration'] = TR
                image_final.header._structarr['pixdim'][4] = TR
                image_final.to_filename('data/simulazione_results/fmri/{}{}.nii'.format(fname+fnamer, trialn))

        # Save movemet offsets
            if save_motion:
                np.savetxt('data/simulazione_results/motionreg/movement_offs_run{}{}.1D'.format(r+1, trialn), movement_offsets, delimiter=' ')
            

        else:
            if save:
                fnamer+='_run{}'.format(r+1)
                image_final = image.new_img_like(data, run_zscore, affine=data.affine, copy_header=True)
                image_final.header._structarr['slice_duration'] = TR
                image_final.header._structarr['pixdim'][4] = TR
                image_final.to_filename('data/simulazione_results/fmri/{}{}.nii'.format(fname+fnamer, trialn))
        
        idx+=run_len

    print('Done with: all. It took:    '.format(r+1), time.time() - tstart, '  seconds')
    
    sys.stdout = orig_stdout
    f.close()

    print('finished')
    