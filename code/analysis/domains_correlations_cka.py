import numpy as np
import pandas as pd
from canonical_correlation_funcs import canonical_correlation
import matplotlib.pyplot as plt
import seaborn as sns

from permutation_schema_func import permutation_schema

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
   
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(results, ax=ax, vmin=0, vmax=1, square=True, annot=True, xticklabels=domains, yticklabels=domains, cmap='rocket_r', cbar_kws=dict(pad=0.01,shrink=0.9, label=measure))
    
    # Labels and layout
    ax.set_title('{} between domains'.format(measure))
    plt.tight_layout()

    # Save
    plt.savefig(path + 'domains_corr_{}.png'.format(measure))

def barplot_corrs(results, domains, measures_names, path):
   
    # Parameters
    measures = results.shape[0]
    n_domains = results.shape[1]
    bar_width = 0.2

    x = np.arange(n_domains)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i in range(measures):
        ax.bar(x + i * bar_width, results[i], bar_width, label=measures_names[i])

    # Labels and layout
    ax.set_xlabel('Domains')
    ax.set_ylabel('Similarity')
    ax.set_title('Average similarity between full model and shuffled model for each domain')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(domains, rotation=45)
    ax.set_ylim(0.5,1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Legend
    ax.legend(title='Similarity measures')

    # Save
    plt.savefig(path + 'similarity_fullvsshuffledmodels.png')

if __name__ == '__main__':
    
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'


    if 1+1 == 3:
      
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
      model_group = pd.read_csv(path + 'group_conv_ds.csv', sep=',')
          
      # Init results
      CKA_res = np.zeros((len(domains), len(domains)))
      dbCKA_res = np.zeros((len(domains), len(domains)))
      CCA_res = np.zeros((len(domains), len(domains)))
      
      for d1, dom1 in enumerate(domains.keys()):
          domain1 = model_group[domains[dom1]].to_numpy()
          
          for d2, dom2 in enumerate(domains.keys()):
            domain2 = model_group[domains[dom2]].to_numpy()
          
            CKA_res[d1, d2] = linear_cka(domain1, domain2)
            dbCKA_res[d1, d2] = feature_space_linear_cka(domain1, domain2, debiased=True)
            CCA_res[d1, d2] = canonical_correlation(domain1, domain2)[1] #Adjusted R2
          
      
      plot_corrs(CKA_res, list(domains.keys()), 'CKA', path)
      plot_corrs(dbCKA_res, list(domains.keys()), 'CKA_debiased', path)
      plot_corrs(CCA_res, list(domains.keys()), 'CCA', path)


    # Load task and Create Full model
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    full_model = {'full_model': np.hstack([domains[d] for d in domains_list])}
    full = full_model['full_model']
    domains['full'] = full

    # Initialize arrays to save all permuted results 
    CCA_shuff_full_res = np.full((50, len(domains)), np.nan)
    CCA_full_shuff_res = np.full((50, len(domains)), np.nan)
    CKA_res = np.full((50, len(domains)), np.nan)
    dbCKA_res = np.full((50, len(domains)), np.nan)
  
    for d, domain in enumerate(list(domains.keys())):
      # Create permutation schema to shuffle columns of left-out domain for variance partitioning
      vperm_schema = permutation_schema(domains[domain].shape[0], n_perms=50, chunk_size=1)

      # For each permutation, build full model and shuffle columns of left-out domain
      for vperm in range(1, vperm_schema.shape[1]):
          vorder = vperm_schema[:, vperm] # first colum contains original order (non-permuted)                 
          shuffled = np.hstack([domains[dom] if dom!=domain else domains[dom][vorder,:] for dom in domains.keys()])
          
          # Run cca 
          CCA_shuff_full_res[vperm-1, d], _, _, _, _, _, _ = canonical_correlation(shuffled, full)
          CCA_full_shuff_res[vperm-1, d], _, _, _, _, _, _ = canonical_correlation(full, shuffled)
          
          # Run CKA
          dbCKA_res[vperm-1, d] = feature_space_linear_cka(shuffled, full, debiased=True)
          CKA_res[vperm-1, d] = linear_cka(shuffled, full)
          
    # Get average R2 results across perms
    CCA_XY_res_avg = np.mean(CCA_shuff_full_res, axis=0)
    CCA_YX_res_avg = np.mean(CCA_full_shuff_res, axis=0)
    CKA_res_avg = np.mean(CKA_res, axis=0)
    dbCKA_res_avg = np.mean(dbCKA_res, axis=0)

    labels = [dom + ' ' + str(domains[dom].shape[1]) for dom in list(domains.keys())]

    barplot_corrs(np.vstack((CCA_XY_res_avg, CCA_YX_res_avg, CKA_res_avg)), labels, ['CCA shuffled vs full', 'CCA full vs shuffled', 'CKA'], path)

    print('d')