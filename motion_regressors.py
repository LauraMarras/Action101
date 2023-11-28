import numpy as np
from random import randint
from scipy.stats import zscore
from scipy.ndimage import zoom, affine_transform
#from skimage import transform
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
        trend_scaled = signal_fit*scaling(p)
        signal_final[:,p] = trend_scaled[window_size:(nTRs+window_size)]
    
    return signal_final


def rotate_mri(mriVolume, upscale, movement_offsets):

    mriVolume_rot_dis_res=[]
    x,y,z,t = mriVolume.shape()
    #mriVolume = mriVolume.astype("uint16")
   
    # resize volume 
    mriVolume_res = zoom.resize(mriVolume, upscale, mode='nearest')

    # apply rotation parameters
    mriVolume_rot_dis = affine_transform(mriVolume_res, movement_offsets[0:3],movement_offsets[3:6])

    # scale down to original resolution
    #mriVolume_rot_dis_res=imresize3(mriVolume_rot_dis,size(mriVolume), 'nearest');
    #mriVolume_rot_dis_res=mriVolume_rot_dis_res([1:size(mriVolume,1)],[1:size(mriVolume,2)],[1:size(mriVolume,3)]);

    return mriVolume_rot_dis_res