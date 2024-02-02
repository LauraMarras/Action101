import numpy as np
import os
from scipy.ndimage import zoom, affine_transform
from skimage import transform
from utils.exectime_decor import timeit

def get_movoffset_fromdata(nTRs, regressors_path, dimensions=(2,2,3), seed=0):
    
    """
    Generate movement offsets signal along time starting from real data

    Inputs:
    - nTRs : int, number of TRs of wanted signal
    - regressors_path : str, path where real data movement regressors are stored
    - dimensions : tuple, dimension of voxels in mm, used to scale offsets; default = (2,2,3)
    - seed : int, seed for random selection of motion regressors real data; default = 0

    Outputs:
    - offset_signals_scaled : array, 2d matrix of shape nTRs by n_dims*2, signal for each movement offset, scaled by voxel dimensions
    - offset_signals : array, 2d matrix of shape nTRs by n_dims*2, signal for each movement offset, not scaled
    """

    # Load movement regressors of real subjects
    sublist = os.listdir(regressors_path)

    # Set seed
    np.random.seed(seed)

    # Initialize offset array
    offset_signals = np.full((nTRs, len(dimensions)*2), np.nan)

    # Concatenate movement regressors of multiple subjects (if needed) until reaching run length
    temp = nTRs
    c = 0
    while temp > 0:
        sub = np.genfromtxt(regressors_path + sublist.pop(np.random.randint(0, len(sublist))) + '/derivatives/rest_mocopar.1D')
        idx = np.min((temp, sub.shape[0]))

        sub = sub - sub[0,:]         

        if c > 0:
            lastrow = offset_signals[c-1,:]
            offset_signals[c:idx+c, :] = sub[:idx,:] +lastrow

        else:
            offset_signals[c:idx+c, :] = sub[:idx,:]

        c+=len(sub)
        temp -= len(sub)
    
    # Scale by voxel dimensions
    offset_signals_scaled = offset_signals / np.array([1,1,1, dimensions[0], dimensions[1], dimensions[2]])

    return offset_signals_scaled, offset_signals

def affine_transformation(volume, movement_offsets, upscalefactor=1):
    
    """
    Applies affine transform to MRI volume given rotation and traslation offsets

    Inputs:
    - volume : array, 3d matrix of shape x by y by z, MRI volume at specific timepoint
    - movement_offsets : array, 1d array of shape 6 (3 rotation and 3 traslation)
    - upscalefactor : int, factor to which upscale image, upscalefactor == 1 means no upscaling; default = 1
   
    Outputs:
    - trans_volume : array, d matrix of shape x by y by z, transformed MRI volume
    """

    
    # Upsample volume
    if upscalefactor != 1:
        volume = zoom(volume, upscalefactor, mode='nearest', order=0)

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

    # Scale down to original resolution
    if upscalefactor != 1:
        trans_volume = zoom(trans_volume, 1/upscalefactor, mode='nearest', order=0)

    return trans_volume

@timeit
def add_motion(data_run, dimensions, regressors_path, upscalefactor=1, seed=0, save=None, sub=0, run=0):
    
    """
    Generate motion regressors and transform signal for each voxel

    Inputs:
    - data :  array, 4d matrix of shape x by y by z by time, fMRI signal of single run
    - dimensions : tuple, dimension of voxels in mm, used to scale offsets; default = (2,2,3)
    - regressors_path : str, path where real data movement regressors are stored
    - upscalefactor : int, factor to which upscale image if needed, upscalefactor == 1 means no upscaling; default = 1
    - seed : int, seed for random selection of motion regressors real data; default = 0
    - save : str, path where (whether) to save motion regressors as 1D file; default = None (don't save)
    - sub : int, subject number to save data; default = 0
    - run :  int, run number to save data; default = 0
    
    Outputs:
    - run_motion : array, 4d matrix of shape x by y by z by nTRs, containing fMRI signal with added motion for each voxel
    
    Calls:
    - get_movoffset_fromdata()
    - affine_transformation()
    """
    
    # Get movement offsets
    movement_offsets, mov_offs_notscaled = get_movoffset_fromdata(data_run.shape[-1], regressors_path, dimensions=dimensions, seed=seed)
    
    # Initialize final matrix
    run_motion = np.full(data_run.shape, np.nan)
    
    # Apply affine transform to each timepoint
    for t in range(data_run.shape[-1]):
        run_motion[:,:,:, t] = affine_transformation(data_run[:,:,:,t], movement_offsets[t,:], upscalefactor=upscalefactor)
    
    # Save movemet offsets --> save non scaled
    if save:
        folder_path = 'data/simulazione_results/sub-{}/motionreg/'.format(sub)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.savetxt('{}movement_offs_run{}.1D'.format(folder_path, run), mov_offs_notscaled, delimiter=' ')
            
    return run_motion