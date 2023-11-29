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


def rotate_mri(mriVolume, upscale, movement_offsets):

    mriVolume_rot_dis_res=[]
    x,y,z = mriVolume.shape
    #mriVolume = mriVolume.astype("uint16")
    
    tstart = time.time()
    # Upsample volume
    mriVolume_res = zoom(mriVolume, upscale, mode='nearest', order=0)
    print(time.time() - tstart)
    
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

    tloop = time.time()
   # pad = np.max((np.abs(trans_coords.max() - trans_coords.shape[0]), np.abs(trans_coords.min())))
    pad = np.max(np.concatenate((np.abs(np.max(trans_coords, (0,1,2)) - (np.array(trans_coords.shape[:3])-1)), np.abs(np.min(trans_coords, (0,1,2))))))
    mripadded = np.pad(mriVolume_res, pad, mode='constant')
    trans_coords = trans_coords+pad
    final = np.full(mriVolume_res.shape, np.nan)
    
    x = trans_coords[:,:,:,0]
    y = trans_coords[:,:,:,1]
    z = trans_coords[:,:,:,2] 

    final = mripadded[x,y,z] 
    
    print(time.time() - tloop)

    tresize2 = time.time()
    # scale down to original resolution
    mriVolume_rot_dis_res = zoom(final, 1/upscale, mode='nearest', order=0)
   # mriVolume_rot_dis_res = mriVolume_rot_dis_res([1:size(mriVolume,1)],[1:size(mriVolume,2)],[1:size(mriVolume,3)]);
    print(time.time() - tresize2)

    return mriVolume_rot_dis_res


if __name__ == '__main__':

    nTRs = 260
    scaling = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)
    window_size = 3
   # movement_offsets = generate_movement_regressors(nTRs, scaling, window_size)
    movement_offsets = np.array([0,0,90,0,0,0]) 
    
    data = image.load_img('datasets/run1_template.nii')
    data_map = data.get_fdata()
    data_map = data_map.astype('float32')
    
    # Upsample volume
    tstart = time.time()
    mriVolume_res = zoom(data_map[:,:,:,:10],[2,2,2,1], mode='nearest', order=0)
    print(time.time() - tstart)
    volume = mriVolume_res[:,:,:,0]


    translated = rotate_mri(volume, 6, movement_offsets)
    
    print('d')

