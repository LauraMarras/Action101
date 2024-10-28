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
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

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
      normal_y = np.sqrt(normal_y**2 - ((n/(n-2)) * np.dot(SS_y, SS_y)) + ((Snorm_y * Snorm_y) / bias_correction_factor))

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

# This is taken from Simon Kornblith's CKA tutorial notebook on Google 
# https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)

def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)
     