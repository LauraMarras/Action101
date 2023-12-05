from transforms3d import affines, euler
from skimage import transform
import numpy as np

T = [20, 30, 40] # translations
R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
Z = [1,1,1] # zooms
A = affines.compose(T, R, Z)
rots = np.array([90, 45, 0])
trans = np.array([20, 10, 0])
rottrans = transform.SimilarityTransform(rotation=np.radians(rots), translation = trans)
rottrans = transform.EuclideanTransform(rotation=np.radians(rots), translation=trans)



R = euler.euler2mat(1,0,0, 'szyx')


print('d')