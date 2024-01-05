import numpy as np
from nilearn import image
from scipy.stats import zscore, norm
from scipy.ndimage import zoom, affine_transform
from skimage import transform
import pandas as pd
import time
from matplotlib import pyplot as plt
import sys
from sklearn.mixture import GaussianMixture
import os

def convolve_HRF(mat, TR=2, hrf_p=8.6, hrf_q=0.547, dur=12):
    
    """
    Convolve each column of matrix with a HRF 
    
    Inputs:
    - mat : matrix to be convolved
    - TR : sampling interval in seconds (fMRI TR)
    - hrf_p : parameter of HRF
    - hrf_q : parameter of HRF
    - dur : duration of HRF, in seconds

    Outputs:
    - group_convolved : dataframe of full group model convolved
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

def affine_transformation(volume, movement_offsets, upscalefactor=1, printtimes=False):
    
    """
    Applies affine transform to MRI volume given rotation and traslation offsets

    Inputs:
    - volume : original MRI volume, matrix of shape x by y by z
    - movement_offsets : movement offsets, array of shape 6 (3 rotation and 3 traslation)
    - upscalefactor : factor to which upscale image, int, upscalefactor == 1 means no upscaling; default = 1
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
    - save : filename_suffix to save figure, if want to save; default = None
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

def segment(volume, n_tissues=4, use_threshold=False, plot=False, save=None):
    
    """
    Segment volume into n populations of voxels

    Inputs:
    - volume : array, original MRI volume, matrix of shape x by y by z
    - n_tissues : int, number of populations to be segmented; default = 4 (air, white matter, grey matter, csf)
    - use_threshold : bool, whether to not consider voxels supposedly located outside of brain based on threshold; default = False
    - plot : bool, whether to plot histogram and gaussian of extracted populations; default = False
    - save : str, whether (filename_suffix) to save tissues mask; default = None
    Outputs:
    - mat_max : array, matrix of shape x by y by z indicating membership for each voxel
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

def create_trend(nTRs, volume, tissues_mask, seed=0, TR=2, save=None):
    
    """
    Generate trend signal for each voxel

    Inputs:
    - nTRs : int, duration of run in TR
    - volume : array, original MRI volume, matrix of shape x by y by z
    - tissues_mask : array, matrix of shape x by y by z indicating membership for each voxel (air, white matter, grey matter, csf)
    - seed : int, seed for random generation; default = 0
    - TR : int, default = 2
    - save : str, whether (filename_suffix) to save polynomial coefficients; default = None
    
    Outputs:
    - trend : array, matrix of shape x by y by z by nTRs containing trend timeseries for each voxel
    - poly_coeff_mat : array, matrix of shape x by y by z indicating trend polynomial coefficients for each voxel
    """
    
    # Get number of tissues
    tissues = np.unique(tissues_mask)

    # Esitmate degree of the polynomial based on legth of each run
    poly_deg = 1 + round(TR*nTRs/150)

    # Initialize matrix of trend for each voxel get a time series of trends
    trend = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], nTRs))
    poly_coeff_mat = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], poly_deg+1))
    
    # Set seed
    np.random.seed(seed)

    # Generate random coefficients: matrix of poly coefficients (based on polydeg) for each tissue
    random_coeffs = np.random.randn(poly_deg+1, len(tissues))

    # Scale the random coeffients according to order and tissue mean intensity
    poly_sorted = np.sort(random_coeffs, axis=1)
    
    scale_poly_order = np.append(np.power(10, np.arange(2,(poly_deg)*2 +1,2))[::-1], 1)
    scale_tissue = np.array([np.mean(volume[np.where(tissues_mask == tissue)]) for tissue in tissues])/np.mean(volume)/100
    
    poly_scaled = poly_sorted / scale_poly_order[:, None] * scale_tissue

    # Create trend time series for each tissue and assign to matrix using tissue mask
    for tissue in tissues:
        trend[tissues_mask == tissue, :] = np.round(np.polyval(poly_scaled[:,tissue], np.arange(nTRs)))
        poly_coeff_mat[tissues_mask == tissue, :] = poly_scaled[:,tissue]
            
    # Save polynomial coefficients as nifti file
    if save:
        save_images(poly_coeff_mat, save)

    return data_run+trend

def get_motion_offsets_data(nTRs, path_reg, dimensions=(2,2,3), seed=0):
    
    """
    Generate movement offsets signal along time starting from real data

    Inputs:
    - nTRs : int, number of TRs of wanted signal
    - dimensions : tuple, dimension of voxels in mm, used to scale offsets; default = (2,2,3)
    - seed : int, seed for random generation; default = 0
    Outputs:
    - offset_signals : signal for each movement offset, matrix of shape nTRs by n_dims*2
    """

    # Load movement regressors of real subjects
    sublist = os.listdir(path_reg)

    # Set seed
    np.random.seed(seed)

    # Randomly pick 3 subjects
    subs = np.random.randint(0, len(sublist), 3)

    # Initialize offset array
    offset_signals = np.full((nTRs, len(dimensions)*2), np.nan)

    # 
    temp = nTRs
    c = 0
    for s in subs:
        sub = np.genfromtxt(path_reg + sublist[s] + '/derivatives/rest_mocopar.1D')
        idx = np.min((temp, sub.shape[0]))

        sub = sub - sub[0,:]         

        if c > 0:
            lastrow = offset_signals[c-1,:]
            offset_signals[c:idx+c, :] = sub[:idx,:] +lastrow

        else:
            offset_signals[c:idx+c, :] = sub[:idx,:]

        c+=len(sub)
        temp -= len(sub)

        if temp <= 0:
            break
    
    # scale
    offset_signals = offset_signals / np.array([1,1,1, dimensions[0], dimensions[1], dimensions[2]])
    return offset_signals

def generate_noise(data_signal, noise_level=4, TR=2, seed=0, save=None):
    
    (x, y, slices, n_points) = data_signal.shape
    
    # Set seed
    np.random.seed(seed)

    # Create gaussian noise
    noise = np.random.randn(x, y, slices, n_points)*noise_level

    # Convolve noise with HRF
    noise_conv = convolve_HRF(noise, TR)

    # Save noise
    if save:
        save_images(noise_conv, save)

    return data_signal + noise_conv

def downsample_timeshift(task, slices, task_time_res=0.05, TR=2):
    
    # Get dimensions
    n_points = int(task.shape[0]/TR*task_time_res)

    # Downsample convolved regressors back to TR resolution and add timeshift for each slice
    task_downsampled_byslice = np.full((slices, n_points, task.shape[1]), np.nan)
    for t in range(n_points):
        for s in range(slices):
            task_downsampled_byslice[s,t,:] = task[int(t*TR/task_time_res) + s]
    
    return task_downsampled_byslice
    
def seminate_mask(task, mask, SNR=5, seed=0):
    # Set seed
    np.random.seed(seed)
    
    # Initialize data matrix
    data_signal = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], task.shape[1]))

    # Create signal by multiply task data by random betas
    betas = np.random.randn(task.shape[2])
    signal = np.dot(task, betas)
    
    # Get mask indices
    (x_inds, y_inds, z_inds) = np.where(mask == 1)

    # Add noise (scaled by SNR factor) and assign signal to each voxel within mask
    noise = np.random.randn(x_inds.shape[0]) + SNR
    data_signal[x_inds, y_inds, z_inds, :] = (signal[z_inds].T*noise).T

    return data_signal

def generate_motion(data, dimensions, regressors_path, upscalefactor=1, seed=0, save=None):
    
    # Get movement offsets
    movement_offsets = get_motion_offsets_data(data.shape[-1], regressors_path, dimensions=dimensions, seed=seed)
    
    # Initialize final matrix
    run_motion = np.full(data.shape, np.nan)
    
    # Apply affine transform to each timepoint
    for t in range(data.shape[-1]):
        run_motion[:,:,:, t] = affine_transformation(data[:,:,:,t], movement_offsets[t,:], upscalefactor=upscalefactor)
    
    # Save movemet offsets
    if save:
        np.savetxt('data/simulazione_results/{}.1D'.format(save), movement_offsets, delimiter=' ')
            
    return run_motion

def save_images(img_tosave, path):
    img = image.new_img_like(data_nii, img_tosave, affine=data_nii.affine, copy_header=True)
    img.to_filename('data/simulazione_results/{}.nii'.format(path))
    

if __name__ == '__main__':
    tstart = time.time()

    # Define options
    n_subs = 1
    add_noise = True
    add_trend = True
    add_motion = True
    
    # Saving options
    save = True
    filename_prefix = 'simul'
    filename_suffix = 'debug2'

    # Print output to txt file
    orig_stdout = sys.stdout
    logfile = open('data/simulazione_results/logs/out{}.txt'.format(filename_suffix), 'w')
    sys.stdout = logfile

    # Define fMRI parameters
    TR = 2
    SNR = 5
    noise_level = 4
    n_tissues = 4 # air, white matter, grey matter, csf

    # Movement parameters
    movement_upscale = 1
    regressors_path = 'data/simulazione_datasets/motionreg/'
    
    # Load task data
    data_path = 'data/models/Domains/group_us_conv_'
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
        data_nii = image.load_img('data/simulazione_datasets/sub{}/run1_template.nii'.format(sub+1))
        mask_nii = image.load_img('data/simulazione_datasets/sub{}/atlas_2orig.nii'.format(sub+1))

        fmri_data = data_nii.get_fdata()[:,:,:,0] # Get single volume
        semination_mask = mask_nii.get_fdata()

        # Get n_voxels and slices, mean and std
        voxel_dims_mm = tuple(data_nii.header._structarr['pixdim'][1:4])
        x,y,slices,_ = data_nii.shape

        data_avg = np.mean(data_nii.get_fdata(), axis=3) # problema se carichiamo solo un volume!!
        data_std = np.std(data_nii.get_fdata(), axis=3, ddof=1)

        print('Sub {}. Done with: loading data, defining parameters. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Downsample convolved regressors back to TR resolution and add timeshift for each slice    
        task_downsampled_byslice = downsample_timeshift(task, slices, task_time_res, TR)
        print('Sub {}. Done with: downsampling and adding timeshift. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Create fMRI signal starting from task and seminate only in mask
        data_signal = seminate_mask(task_downsampled_byslice, semination_mask, SNR, seed_schema[sub,-1])
        print('Sub {}. Done with: creating fMRI signal from task. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Generate and add noise
        if add_noise:
            filename_prefix += '_noise'
            data_signal = generate_noise(data_signal, noise_level, TR, seed_schema[sub,-2], save='sub{}/noise/noise_{}'.format(sub+1, filename_suffix)) #save=filename_suffix
        print('Sub {}. Done with: generating and adding noise. It took:  {}  seconds'.format(sub+1, time.time() - tstart))

        # Segment
        tissues_mask = segment(fmri_data, n_tissues, use_threshold=False, plot=False, save='sub{}/mask/mask_{}'.format(sub+1, filename_suffix)) #save=None
        print('Sub {}. Done with: segmenting. It took:  {}  seconds'.format(sub+1, time.time() - tstart))
        
        # Iterate over runs
        for r in range(n_runs):

            # Set filename for each run
            filename_prefixr = ''
            filename_suffixr = '_run{}_{}'.format(r+1, filename_suffix)

            # Get data of single run
            run_len = run_dur_TR[r]
            
            run_idx = [*range(run_cuts[r][0], run_cuts[r][1])]
            data_run = data_signal[:,:,:,run_idx]

            # Generate Trend (for each run separately)
            if add_trend:
                filename_prefixr += '_trend' 
                data_run = create_trend(run_len, fmri_data, tissues_mask, seed=seed_schema[sub,r], save='sub{}/trend/polycoeffs{}'.format(sub+1, filename_suffixr)) # save=None
            print('Sub {}. Done with: generating trend for run {}. It took:  {}  seconds'.format(sub+1, r+1, time.time() - tstart))
                
            # Zscore
            run_zscore = zscore((data_run), axis=3, nan_policy='omit')
            data_run = run_zscore * np.expand_dims(data_std, axis=3) + np.expand_dims(data_avg, axis=3) #problema se carichiamo solo un volume
            print('Sub {}. Done with: zscoring for run {}. It took:  {}  seconds'.format(sub+1, r+1, time.time() - tstart))

            # Add motion
            if add_motion:        
                filename_prefixr += '_motion'
                data_run = generate_motion(data_run, dimensions=voxel_dims_mm, upscalefactor=movement_upscale, regressors_path=regressors_path, seed=seed_schema[sub, n_runs+r], save='sub{}/motionreg/movement_offs{}'.format(sub+1, filename_suffixr)) # save=None
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