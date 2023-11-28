import numpy as np
from random import randint
from scipy.stats import zscore
from scipy.ndimage import zoom, affine_transform
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
   
    # resize volume 
    #mriVolume_res = zoom(mriVolume, upscale, mode='nearest')

    rot_matx = np.identity(4)
    rot_matx[1:3, 1:3] = np.array([[np.cos(movement_offsets[0]), np.sin(movement_offsets[0])],[-np.sin(movement_offsets[0]), np.cos(movement_offsets[0])]])
    
    rot_maty = np.identity(4)
    rot_maty[(0,0)] = np.cos(movement_offsets[1])
    rot_maty[(0,2)] = -np.sin(movement_offsets[1])
    rot_maty[(2,0)] = np.sin(movement_offsets[1])
    rot_maty[(2,2)] = np.cos(movement_offsets[1])

    rot_matz = np.identity(4)
    rot_matz[0:2, 0:2] = np.array([[np.cos(movement_offsets[2]), -np.sin(movement_offsets[2])],[np.sin(movement_offsets[2]), np.cos(movement_offsets[2])]])
    
    rot_mat = np.matmul(rot_matx, rot_maty, rot_matz)

    rot_mat[:-1, -1] = movement_offsets[-3:]
    
    transformed_mat = transform.AffineTransform(rot_mat)
    coords = transform.warp(mriVolume, transformed_mat)
    
    # # scale down to original resolution
    # #mriVolume_rot_dis_res=imresize3(mriVolume_rot_dis,size(mriVolume), 'nearest');
    # #mriVolume_rot_dis_res=mriVolume_rot_dis_res([1:size(mriVolume,1)],[1:size(mriVolume,2)],[1:size(mriVolume,3)]);

    return coords


if __name__ == '__main__':

    nTRs = 260
    scaling = np.concatenate(((np.random.randn(3)/10), (np.random.randn(3)/3)), 0)
    window_size = 3
    movement_offsets = generate_movement_regressors(nTRs, scaling, window_size)

    
    data = image.load_img('datasets/run1_template.nii')
    data_map = data.get_fdata()
    volume = data_map[:,:,:,0]

    translated = rotate_mri(volume, 6, movement_offsets[0])
    
    print('d')

