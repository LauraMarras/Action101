import numpy as np
from scipy.stats import zscore
from scipy.ndimage import zoom
from nilearn import image
from skimage import transform
import pandas as pd
import time

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

    # Create rotation matrix
    angles = np.radians(movement_offsets[:3])
    shift = - np.array(volume.shape)/2 #Add shift to move origin to the center
    rotation = transform.SimilarityTransform(rotation=angles, translation=shift, dimensionality=3)
    r = transform.SimilarityTransform(rotation=angles, dimensionality=3)
    s = transform.SimilarityTransform(translation=shift, dimensionality=3)

    # Create translation matrix
    displacement = movement_offsets[3:]
    translation = transform.SimilarityTransform(translation=displacement-shift, dimensionality=3) # Shift back to origin
    t = transform.SimilarityTransform(translation=displacement, dimensionality=3)

    # Compose transforms by multiplying their matrices
    matrix = translation.params @ rotation.params
    m = t.params @ np.linalg.inv(s.params) @ r.params @ s.params

    # Apply transforms to coordinates
    coords = np.rollaxis(np.indices(volume.shape), 0, 1+volume.ndim)
    coords = np.append(coords, np.ones((coords.shape[0], coords.shape[1], coords.shape[2], 1)), axis=3)
    trans_coords = np.dot(coords, np.linalg.inv(matrix).T)
    t_c = np.dot(coords, np.linalg.inv(m).T)
    trans_coords = np.delete(trans_coords, 3, axis=3).astype(int)
    t_c = np.delete(t_c, 3, axis=3).astype(int)

    # Add padding to original volume
    # pad = np.max((np.abs(trans_coords.max() - trans_coords.shape[0]), np.abs(trans_coords.min())))
    # pad = np.max(np.concatenate((np.abs(np.max(trans_coords, (0,1,2)) - (np.array(trans_coords.shape[:3])-1)), np.abs(np.min(trans_coords, (0,1,2))))))
    pad = np.max(volume.shape)
    mripadded = np.pad(volume, pad, mode='constant')
    trans_coords = trans_coords+pad
    t_c = t_c+pad
    
    x = trans_coords[:,:,:,0]
    y = trans_coords[:,:,:,1]
    z = trans_coords[:,:,:,2] 

    x1 = t_c[:,:,:,0]
    y1 = t_c[:,:,:,1]
    z1 = t_c[:,:,:,2] 

    # final = np.full(volume.shape, np.nan)
    final = mripadded[x,y,z] 
    final1 =  mripadded[x1,y1,z1]
    
    # scale down to original resolution
    if upscale:
        final = zoom(final, 1/upscale, mode='nearest', order=0)
  
    return final, final1


if __name__ == '__main__':

    nTRs = 260
    scaling = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)
    window_size = 3
   # movement_offsets = generate_movement_regressors(nTRs, scaling, window_size)
    movement_offsets = np.array([0,0,90,0,0,0]) 
    
    data = image.load_img('data/simulazione_datasets/run1_template.nii')
    data_map = data.get_fdata()
    data_map = data_map.astype('float32')
    
    # # Upsample volume
    # tstart = time.time()
    # mriVolume_res = zoom(data_map[:,:,:,:10],[2,2,2,1], mode='nearest', order=0)
    # print(time.time() - tstart)
    # volume = mriVolume_res[:,:,:,0]


    final, final1 = rotate_mri(data_map[:,:,:,0], movement_offsets)
    
    print('d')

