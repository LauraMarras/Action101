import numpy as np
import os
from matplotlib import pyplot as plt
from nilearn import image
from utils.exectime_decor import timeit

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

def autocorr_diff(data_noise, data_noise_conv, coords=(0,0,0), path=''):
    
    """
    Measure and plot autocorrelation of two voxels 
    
    Inputs:
    - data_noise : array, 4d matrix of shape x by y by z by time
    - data_noise_conv : array, 4d matrix of shape x by y by z by time
    - coords : tuple, sequence of 3 int indicating x, y and z coordinates of chosen voxel; default = (0,0,0)
    - path : str, path where (whether) to save measured difference in autocorrelation pre and post convolution or not; default = ''
    
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

    # Save figure
    plt.savefig('{}autocorrelation_{}.png'.format(path, str(coords)))

@timeit
def add_noise(data_signal, noise_level=4, TR=2, seed=0, sub=0, reference=None, check_autocorr=None):
    
    """
    Add noise to fMRI data matrix

    Inputs:
    - data_signal : array, 4d matrix of shape = x by y by z by time containing task-related signal only in voxels within mask, else zeros or containing only zeros
    - noise_level : int or float, indicates scale of gaussian noise; default = 4
    - TR : int or float, fMRI resolution, in seconds; default = 2
    - seed : int, seed for random generation of gaussian noise; default = 0
    - sub : int, subject number to save data; default = 0
    - reference : niilike object, used as reference to save data as nifti file, if None don't save; default = None
    - check_autocorr : dict, contains 'path' where (whether) to save measured difference in autocorrelation pre and post convolution or not, and 'coords' of voxel to measure; default = None (don't measure)
    
    Outputs:
    - data_noise : array, 4d matrix of shape = x by y by slices containing HRF-convoluted noise and task-related signal
    
    Calls:
    - convolve_HRF()
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
    if reference:
        folder_path = 'data/simulazione_results/sub-{}/noise/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img = image.new_img_like(reference, noise_conv, affine=reference.affine, copy_header=True)
        img.to_filename('{}noise.nii'.format(folder_path))
    
    # Check difference in autocorrelation
    if check_autocorr:
        autocorr_diff(data_signal+noise, data_noise, coords=check_autocorr[check_autocorr['coords']], path=check_autocorr['path'])

    return data_noise