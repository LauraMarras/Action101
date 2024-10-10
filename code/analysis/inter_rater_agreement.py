import numpy as np


def feature_space_linear_cka(features_x, features_y):
    
    """
    Compute CKA with a linear kernel, in feature space.
    Inputs:
    - features_x: array, 2d matrix of shape = samples by features
    - features_y: 2d matrix of shape = samples by features
    
    Output:
    - the value of CKA between X and Y
    """

    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    return dot_product_similarity / (normalization_x * normalization_y)

if __name__ == '__main__':

    fakeX = np.array([[1,0], [2,1]])
    fakeY = np.array([[4,6,3,8,1,6,7,1,2,0,7,8], [6,3,8,2,8,4,4,7,9,3,9,1]])
    res = feature_space_linear_cka(fakeX, fakeY)   

    print('d')