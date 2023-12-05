import numpy as np
from scipy.stats import zscore
from scipy.ndimage import zoom
from nilearn import image
from skimage import transform
import pandas as pd
import time
from matplotlib import pyplot as plt

def generate_movement_regressors(nTRs, scaling, window_size=3):

    nParams = len(scaling)

    x = np.arange(0,nTRs+window_size)
    deg = np.random.randint(2,7)
    poly_coeffs = np.random.randn(deg)
    
    signal_fit = np.polyval(poly_coeffs,x)
    signal_fit = zscore(signal_fit) + np.random.randn(len(x))
    signal_fit = signal_fit/np.std(signal_fit)
    signal_series = pd.Series(signal_fit)
    signal_fit = signal_series.rolling(window_size+1).mean()

    signal_final = np.empty((nTRs, nParams))
    for p in range(nParams):
        trend_scaled = signal_fit*scaling[p]
        signal_final[:,p] = trend_scaled[window_size:(nTRs+window_size)]
    
    return signal_final


def rotate_mri(volume, movement_offsets, upscale=False, upscalefactor=6):
    
    tstart = time.time()
    
    # Upsample volume
    if upscale:
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
    trans_coords = np.delete(trans_coords, 3, axis=3).astype(int)

    # Add padding to original volume
    pad = np.max(np.concatenate((np.abs(np.max(trans_coords, (0,1,2)) - (np.array(trans_coords.shape[:3])-1)), np.abs(np.min(trans_coords, (0,1,2))))))
    volume_padded = np.pad(volume, pad, mode='constant')
    trans_coords = trans_coords+pad
    
    # Map from original to transformed volume through transformed coordinates
    x = trans_coords[:,:,:,0]
    y = trans_coords[:,:,:,1]
    z = trans_coords[:,:,:,2] 

    trans_volume = volume_padded[x,y,z] 
    ttransform = time.time() - tupscale

    # Scale down to original resolution
    if upscale:
        trans_volume = zoom(trans_volume, 1/upscalefactor, mode='nearest', order=0)
    tdownscale = time.time() - ttransform

    print('Time to upscale:{}s \nTime to transform:{}s \nTime to downscale:{}s'.format(tupscale, ttransform, tdownscale))
    
    return trans_volume


def plot_transform(original, transformed, off, x=64,y=64,s=19, save=None):
    
    fig, axs = plt.subplots(3,2, gridspec_kw=dict(height_ratios=[128/38, 1, 1], width_ratios=[1,1]))#  sharex=True, sharey=False)
    
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

    # Add Titles
    axs[0,0].set_title('Original Volume')
    axs[0,1].set_title('Transformed Volume')

    axs[0,0].set_ylabel('Axial \n(z={})'.format(s))
    axs[1,0].set_ylabel('Sagittal \n(x={})'.format(x))
    axs[2,0].set_ylabel('Coronal \n(y={})'.format(y))

    
    # Add transformation parameters
    plt.text(0.01, 0.99, 
             'trans x={} \ntrans y={} \ntrans z={} \npitch={} \nroll={} \nyaw={}'.format(off[3], off[4], off[5], off[0], off[1], off[2]),
        verticalalignment='top', horizontalalignment='left',
        transform=plt.gcf().transFigure,
        color='k', fontsize=10)
    

    if save:
        plt.savefig(save)
    else:
        plt.show()

if __name__ == '__main__':

    nTRs = 260
    scaling = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)
    window_size = 3
    movement_offsets = generate_movement_regressors(nTRs, scaling, window_size)[0,:]
   #     movement_offsets = np.array([0,0,0, 0, 0, 10])
    
    data = image.load_img('data/simulazione_datasets/run1_template.nii')
    data_map = data.get_fdata()
    data_map = data_map.astype('float32')
    
    original = data_map[:,:,:,0]
    transformed = rotate_mri(original, movement_offsets)
    
    plot_transform(original, transformed, movement_offsets, 50,50,15, 'prova')


    sl = 19
    xx = 64
    yy = 64

    fig, axs = plt.subplots(3,2)
    axs[0,0].imshow(final[:,:,sl])
    axs[0,1].imshow(data_map[:,:,sl,0])

    axs[1,0].imshow(final[xx,:,:])
    axs[1,1].imshow(data_map[xx,:,:,0])

    axs[2,0].imshow(final[:,yy,:])
    axs[2,1].imshow(data_map[:,yy,:,0])
    
    print('d')

