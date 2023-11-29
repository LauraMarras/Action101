import numpy as np
from random import randint
from scipy.stats import zscore
from scipy.ndimage import zoom, affine_transform
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from nilearn import image
from skimage import transform
import pandas as pd

def generate_movement_regressors(nTRs, scaling, window_size=3):

    nParams = len(scaling)

    x = np.arange(0,nTRs+window_size)
    deg = randint(2,6)
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


def rotate_mri(mriVolume, upscale, movement_offsets):

    mriVolume_rot_dis_res=[]
    x,y,z = mriVolume.shape
    #mriVolume = mriVolume.astype("uint16")
   
    # Upsample volume
    mriVolume_res = zoom(mriVolume, upscale, mode='nearest')

    #rotation matrix
    rotation = transform.SimilarityTransform(rotation=np.radians(movement_offsets[:3]),dimensionality=3)
    shift = transform.SimilarityTransform(translation=-np.array(mriVolume_res.shape[:3])/2,dimensionality=3)
    # translation matrix
    translation = transform.SimilarityTransform(translation=movement_offsets[3:6],dimensionality=3) 

    # Compose transforms by multiplying their matrices
    matrix = translation.params @ np.linalg.inv(shift.params) @ rotation.params @ shift.params

    # apply transforms to coordinates
    coords = np.rollaxis(np.indices(mriVolume_res.shape), 0, len(mriVolume_res.shape)+1)
    coords = np.append(coords,np.ones((coords.shape[0],coords.shape[1],coords.shape[2],1)), axis=3)
    trans_coords = np.matmul(coords, np.linalg.inv(matrix).T) # .T)
    trans_coords = np.delete(trans_coords,3,axis=3).astype(int)

   # pad = np.max((np.abs(trans_coords.max() - trans_coords.shape[0]), np.abs(trans_coords.min())))
    pad = np.max(np.concatenate((np.abs(np.max(trans_coords, (0,1,2)) - (np.array(trans_coords.shape[:3])-1)), np.abs(np.min(trans_coords, (0,1,2))))))
    mripadded = np.pad(mriVolume_res, pad, mode='constant')
    trans_coords = trans_coords+pad
    final = np.full(mriVolume_res.shape, np.nan)
    for x in range(trans_coords.shape[0]):
        for y in range(trans_coords.shape[1]):
            for z in range(trans_coords.shape[2]):
                final[x,y,z] = mripadded[tuple(trans_coords[x,y,z,:])] 


    # scale down to original resolution
    mriVolume_rot_dis_res = zoom(final, 1/upscale, mode='nearest')
   # mriVolume_rot_dis_res = mriVolume_rot_dis_res([1:size(mriVolume,1)],[1:size(mriVolume,2)],[1:size(mriVolume,3)]);
    
    return mriVolume_rot_dis_res


if __name__ == '__main__':

    nTRs = 260
    scaling = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)
    window_size = 3
   # movement_offsets = generate_movement_regressors(nTRs, scaling, window_size)
    movement_offsets = np.array([0,0,90,0,0,0]) 
    
    data = image.load_img('datasets/run1_template.nii')
    data_map = data.get_fdata()
    volume = data_map[:,:,:,0]

    translated = rotate_mri(volume, 6, movement_offsets)
    
    print('d')

