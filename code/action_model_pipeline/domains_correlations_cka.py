import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    
    # Define parameters
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'
    debiasing = True
    convolved = False
    
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
                  'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10', 'target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2', 'sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present',
                  'EBL', 'EIA', 'gesticolare', 'simbolic_gestures','durativity', 'telicity', 'iterativity', 'dinamicity']
    }

    # Load CKA results
    try:
        CKA_df = pd.read_csv('{}domains/CKA{}_btw_domains{}.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'))
    
    except FileNotFoundError:
        
        # Load group model
        model_group = pd.read_csv('{}group_{}_ds.csv'.format(path, 'conv' if convolved else 'bin'), sep=',')
          
        # Initialize results matrix
        CKA_res = np.zeros((len(domains), len(domains)))

        for d1, dom1 in enumerate(domains.keys()):
            domain1 = model_group[domains[dom1]].to_numpy()
            
            for d2, dom2 in enumerate(domains.keys()):
                domain2 = model_group[domains[dom2]].to_numpy()
            
                CKA_res[d1, d2] = linear_cka(domain1, domain2, debiasing=debiasing)
          
        # Save
        CKA_df = pd.DataFrame(CKA_res, columns=domains.keys(), index=domains.keys())
        CKA_df.to_csv('{}domains/CKA{}_btw_domains{}.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'))