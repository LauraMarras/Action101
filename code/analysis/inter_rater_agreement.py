import numpy as np
import pandas as pd
from canonical_correlation_funcs import canonical_correlation
import matplotlib.pyplot as plt


def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
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

def linear_cka(X, Y):
    
    """
    Compute CKA with a linear kernel, in feature space.
    Inputs:
    - X: array, 2d matrix of shape = samples by features
    - Y: 2d matrix of shape = samples by features
    
    Output:
    - CKA: float, the value of CKA between X and Y
    """

    # Recenter
    X = X - np.mean(X, 0, keepdims=True)
    Y = Y - np.mean(Y, 0, keepdims=True)

    # Get CKA between X and Y
    dot_product_similarity = np.linalg.norm(np.dot(X.T, Y)) ** 2 # Squared Frobenius norm of dot product between X transpose and Y
    normalization_x = np.linalg.norm(np.dot(X.T, X)) 
    normalization_y = np.linalg.norm(np.dot(Y.T, Y))

    CKA = dot_product_similarity / np.dot(normalization_x, normalization_y)

    return CKA

def plot_corrs(results, domains, measure, path):
   
    # Parameters
    n_rater_pairs = results.shape[0]
    n_domains = results.shape[1]
    rater_pairs = ['AI vs LM', 'AI vs LT', 'LM vs LT']
    bar_width = 0.2

    x = np.arange(n_domains)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i in range(n_rater_pairs):
        ax.bar(x + i * bar_width, results[i], bar_width, label=rater_pairs[i])

    # Labels and layout
    ax.set_xlabel('Domains')
    ax.set_ylabel('{}'.format(measure))
    ax.set_title('{} between raters for each domain'.format(measure))
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(domains, rotation=45)
    ax.set_ylim(0,1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Legend
    ax.legend(title='Raters')

    # Save
    plt.savefig(path + 'inter-rater_{}_notOHE.png'.format(measure))

if __name__ == '__main__':
    
    taggers = ['LM', 'AI', 'LT']
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'

    domains = {
    'space': ['context_0', 'context_1', 'context_2', 'inter_scale'],
    'movement': ['eff_visibility', 'main_effector_0', 'main_effector_1', 'main_effector_2', 
                 'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6',
                  'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10'],
    'agent_objective': ['target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],
    'social_connectivity': ['sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
    'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
    'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'dinamicity'],
    'full': ['context_0', 'context_1', 'context_2', 'inter_scale', 'eff_visibility', 'main_effector_0', 'main_effector_1', 'main_effector_2', 
                 'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6',
                  'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10', 'target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2', 'sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present', 'EBL', 'EIA', 'gesticolare', 'simbolic_gestures', 'durativity', 'telicity', 'iterativity', 'dinamicity',
                  'EBL', 'EIA', 'gesticolare', 'simbolic_gestures','durativity', 'telicity', 'iterativity', 'dinamicity']
    }
    
    domains_notOHE = {
       'space': ['context', 'inter_scale'], 
       'movement': ['main_effector', 'eff_visibility'], 
       'agent_objective': ['target', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch'], 
       'social_connectivity': ['sociality', 'target', 'touch', 'multi_ag_vs_jointact', 'ToM', 'people_present'], 
       'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'], 
       'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'dinamicity'], 
       'full': ['sociality', 'target', 'touch', 'multi_ag_vs_jointact', 'ToM', 'people_present', 'durativity', 'telicity', 'iterativity', 'dinamicity', 'EBL', 'EIA', 'gesticolare', 'simbolic_gestures', 'agent_H_NH', 'tool_mediated', 'transitivity', 'context', 'inter_scale', 'main_effector', 'eff_visibility']
       }
    
    
    # Load single tagger tagging
    AI = pd.read_csv(path + '{}_ds_notOHE.csv'.format('AI'), sep=',')
    LM = pd.read_csv(path + '{}_ds_notOHE.csv'.format('LM'), sep=',')
    LT = pd.read_csv(path + '{}_ds_notOHE.csv'.format('LT'), sep=',')
    
    # Init results
    CKA_res = np.zeros((3, len(domains)))
    dbCKA_res = np.zeros((3, len(domains)))
    CCA_res = np.zeros((3, len(domains)))

    for d, dom in enumerate(domains.keys()):
        ai = AI[domains_notOHE[dom]].to_numpy()
        lm = LM[domains_notOHE[dom]].to_numpy()
        lt = LT[domains_notOHE[dom]].to_numpy()

        CKA_res[0, d] = linear_cka(ai, lm)
        CKA_res[1, d] = linear_cka(ai, lt)
        CKA_res[2, d] = linear_cka(lm, lt)
        
        dbCKA_res[0, d] = feature_space_linear_cka(ai, lm, debiased=True)
        dbCKA_res[1, d] = feature_space_linear_cka(ai, lt, debiased=True)
        dbCKA_res[2, d] = feature_space_linear_cka(lm, lt, debiased=True)
        
        CCA_res[0, d] = canonical_correlation(ai, lm)[0]
        CCA_res[1, d] = canonical_correlation(ai, lt)[0]
        CCA_res[2, d] = canonical_correlation(lm, lt)[0]
    
    plot_corrs(CKA_res, list(domains.keys()), 'CKA', path)
    plot_corrs(dbCKA_res, list(domains.keys()), 'debiasedCKA', path)
    plot_corrs(CCA_res, list(domains.keys()), 'CCA', path)
    

    print('d')