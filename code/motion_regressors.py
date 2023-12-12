import numpy as np
from scipy.stats import zscore
from scipy.ndimage import zoom
from nilearn import image
from skimage import transform
import pandas as pd
import time
from matplotlib import pyplot as plt
import os

def get_movement_offsets(nTRs, dims=3, window_size=3, seed=0):
    
    """
    Generate movement offsets signal along time

    Inputs:
    - nTRs : int, number of TRs of wanted signal
    - dims : int, dimensionality of volume; default = 3
    - window_size : int, size (number of TRs) of window used for smoothing signal; default = 3
    - seed : seed for the random generation; default = 0

    Outputs:
    - offset_signals : signal for each movement offset, matrix of shape nTRs by dims*2
    """

    # Set seed
    np.random.seed(seed)

    # Set scaling
    scaling = np.concatenate(((np.random.randn(dims)/10), (np.random.randn(dims)/3)), 0)
        
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

def affine_transform(volume, movement_offsets, upscalefactor=6, printtimes=False):
    
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

    # Get volume coordinates 
    coords = np.rollaxis(np.indices(volume.shape), 0, 1+volume.ndim)
    coords = np.append(coords, np.ones((coords.shape[0], coords.shape[1], coords.shape[2], 1)), axis=3) # Add 1s to match shape for multiplication
    
    # Create rotation, shift and translation matrices
    angles = np.radians(movement_offsets[:3])
    shift = - np.array(volume.shape)/2 # shift to move origin to the center
    displacement = movement_offsets[3:]

    r = transform.SimilarityTransform(rotation=angles, dimensionality=3)
    s = transform.SimilarityTransform(translation=shift, dimensionality=3)
    t = transform.SimilarityTransform(translation=displacement, dimensionality=3)
    
    # Compose transforms by multiplying their matrices (mind the order of the operations)
    trans_matrix = t.params @ np.linalg.inv(s.params) @ r.params @ s.params

    # Apply transforms to coordinates
    trans_coords = np.dot(coords, np.linalg.inv(trans_matrix).T)
    trans_coords = np.delete(np.round(trans_coords), 3, axis=3).astype(int)

    # Add padding to original volume
    pad = np.max(np.concatenate((np.abs(np.max(trans_coords, (0,1,2)) - (np.array(trans_coords.shape[:3])-1)), np.abs(np.min(trans_coords, (0,1,2))))))
    volume_padded = np.pad(volume, pad, mode='constant')
    trans_coords = trans_coords+pad
    
    # Map from original to transformed volume through transformed coordinates
    x = trans_coords[:,:,:,0]
    y = trans_coords[:,:,:,1]
    z = trans_coords[:,:,:,2] 

    trans_volume = volume_padded[x,y,z] 
    ttransform = time.time() - tstart

    # Scale down to original resolution
    if upscalefactor != 1:
        trans_volume = zoom(trans_volume, 1/upscalefactor, mode='nearest', order=0)
    tdownscale = time.time() - tstart

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
    off = [round(c,3) for c in movement_offsets]
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


def get_motion_offsets_data(nTRs, path_reg, dimensions=(2,2,3)):
    
    # Load movement regressors of real subjects
    sublist = os.listdir(path_reg)

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



if __name__ == '__main__':

    np.random.seed(0) 
        
    nTRs = 268
   
    data = image.load_img('data/simulazione_datasets/run1_template.nii')
    
    dimensions = data.header._structarr['pixdim'][1:4]
    
    data_map = data.get_fdata().astype('float32')
    
    original = data_map[:,:,:,0]
    
    movement_offsets = get_motion_offsets_data(nTRs, 'data/simulazione_datasets/motionreg/')
    np.savetxt('data/simulazione_results/movement_offsrun1.1D', movement_offsets, delimiter=' ')

    #movement_offsets = get_movement_offsets(nTRs)[0,:]
    #movement_offsets = [0,0,90, 5,5,5]
    # transformed = affine_transform(original, movement_offsets, upscalefactor=1, printtimes=True)
    
    # movement_offsets2 = get_motion_offsets_data(nTRs, 'data/simulazione_datasets/motionreg/')[152,:]
    # transformed2 = affine_transform(original, movement_offsets2, upscalefactor=1, printtimes=True)
    
    # plot_transform(transformed2, transformed, movement_offsets, save='data/simulazione_results/motion_t2')

    print('d')
  



#