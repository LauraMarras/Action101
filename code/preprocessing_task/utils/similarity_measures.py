import numpy as np
from sklearn.cross_decomposition import CCA

def linear_cka(X, Y, debiasing=False):
    
    """
    Compute CKA with a linear kernel, in feature space.
    Inputs:
    - X: array, 2d matrix of shape = samples by features
    - Y: array, 2d matrix of shape = samples by features
    - debiasing: bool, whether to apply debiasing or not; default = False 
    
    Output:
    - CKA: float, the value of CKA between X and Y
    """

    # Recenter
    X = X - np.mean(X, 0, keepdims=True)
    Y = Y - np.mean(Y, 0, keepdims=True)

    # Get dot product similarity and normalized matrices
    similarity = np.linalg.norm(np.dot(X.T, Y))**2 # Squared Frobenius norm of dot product between X transpose and Y
    normal_x = np.linalg.norm(np.dot(X.T, X)) 
    normal_y = np.linalg.norm(np.dot(Y.T, Y))
    
    # Apply debiasing
    if debiasing: 
      n = X.shape[0]
      bias_correction_factor = (n-1)*(n-2)
    
      SS_x = np.sum(X**2, axis=1) # Sum of squared rows 
      SS_y = np.sum(Y**2, axis=1)
      Snorm_x = np.sum(SS_x) # Squared Frobenius norm
      Snorm_y = np.sum(SS_y)
      
      similarity = similarity - ((n/(n-2)) * np.dot(SS_x, SS_y)) + ((Snorm_x * Snorm_y) / bias_correction_factor)
      normal_x = np.sqrt(normal_x**2 - ((n/(n-2)) * np.dot(SS_x, SS_x)) + ((Snorm_x * Snorm_x) / bias_correction_factor)) 
      normal_y = np.sqrt(normal_y**2 - ((n/(n-2)) * np.dot(SS_y, SS_y)) + ((Snorm_x * Snorm_x) / bias_correction_factor))

    # Get CKA between X and Y
    CKA = similarity / np.dot(normal_x, normal_y)

    return CKA

def canonical_correlation(X,Y, center=True):
    
    """
    Performs canonical correlation analysis using sklearn.cross_decomposition CCA package
    
    Inputs:
    - X : array, 2d matrix of shape = n by d1
    - Y : array, 2d matrix of shape = n by d2
    - center: bool, whether to remove the mean (columnwise) from each column of X and Y; default = True

    Outputs:
    - r2 : float, R-squared (Y*B = V)
    - r2adj : float, R-squared (Y*B = V) adjusted
    - A : Sample canonical coefficients for X variables
    - B : Sample canonical coefficients for Y variables
    - r : Sample canonical correlations
    - U : canonical scores for X
    - V : canonical scores for Y
    """

    # Center X and Y
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

    # Canonical Correlation Analysis
    n_components = np.min([X.shape[1], Y.shape[1]]) # Define n_components as the min rank

    cca = CCA(n_components=n_components, scale=True, max_iter=5000)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)

    # Get A and B matrices as done by Matlab canoncorr()
    A = np.linalg.lstsq(X, U, rcond=None)[0]
    B = np.linalg.lstsq(Y, V, rcond=None)[0]

    # Calculate R for each canonical variate
    R = np.full(U.shape[1], np.nan)
    for c in range(U.shape[1]):
        x = U[:,c]
        y = V[:,c]
        r = np.corrcoef(x,y, rowvar=False)[0,1]
        R[c] = r

    # Calculate regression coefficients b_coeffs
    b_coeffs = np.linalg.lstsq(U, V, rcond=None)[0]

    # Predict V using U and b_coeffs
    V_pred = np.dot(U, b_coeffs)

    # Calculate centered predicted Y
    Y_pred = np.dot(V_pred, np.linalg.pinv(B))

    # Calculate R-squared
    SSres = np.sum((Y.ravel() - Y_pred.ravel()) ** 2)
    SStot = np.sum((Y.ravel() - np.mean(Y.ravel())) ** 2)
    r2 = 1 - (SSres / SStot)

    # Adjust by number of X columns
    n = Y_pred.shape[0]
    p = n_components
    r2adj = 1 - (1-r2)*((n-1)/(n-p-1))

    return r2, r2adj, A, B, R, U, V

def rvcoeff(X,Y):
    
    P1= X.shape[1]
    P2 = Y.shape[1]
    N = X.shape[0]

    RV = (P1*P2)/(np.sqrt((P1**2 + (N+1)*P1)*(P2**2 + (N+1)*P2)))

    return RV
