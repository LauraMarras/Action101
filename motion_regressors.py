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
    #mriVolume_res = zoom(mriVolume, upscale, mode='nearest')

    # Rotation
    ## Convert rotation values to radians
    rotation_radians = np.deg2rad(movement_offsets[:3])
    
    ## Create rotation matrices for each axis and combine them
    rotation_matrices = []
    for i in range(3):
        rotation_axis = np.zeros(3)
        rotation_axis[i] = 1  # Set the corresponding axis to 1 for rotation
        rotation = Rotation.from_rotvec(rotation_radians[i] * rotation_axis)
        rotation_matrices.append(rotation.as_matrix())

    ## Combine rotation matrices for each axis
    combined_rotation_matrix = np.eye(4)
    for rot_matrix in rotation_matrices:
        combined_rotation_matrix = np.dot(combined_rotation_matrix, np.vstack((rot_matrix, [0, 0, 0, 1])))


    # Translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = movement_offsets[3:]

    # Combine translation and rotation matrices
    combined_matrix = np.dot(translation_matrix, combined_rotation_matrix)

    # Apply the combined transformation to the original 3D matrix
    transformed_matrix = np.dot(mriVolume, combined_matrix)

    # # scale down to original resolution
    # #mriVolume_rot_dis_res=imresize3(mriVolume_rot_dis,size(mriVolume), 'nearest');
    # #mriVolume_rot_dis_res=mriVolume_rot_dis_res([1:size(mriVolume,1)],[1:size(mriVolume,2)],[1:size(mriVolume,3)]);

    return transformed_matrix


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

