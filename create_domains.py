import pandas as pd
import numpy as np

def get_domain_mat(model_df, domains, path=False, filename=''):
   
    """
    Unpack group model into single domains
    
    Inputs:
    - model_df : dataframe of full group model
    - domains : dictionary containing domain names as keys and list of column names as values
    - path : default = False; path where to save csv files
    - filename : default = '', output filename without extension

    Outputs:
    - matrices : dictionary containing domain names as keys and domain dataframes as values
    """  

    # Create domain matrices
    matrices = {}
    for key, val in domains.items():   
        matrices[key] = model_df[val]
        
        # Save each matrix into csv file
        if path:
            matrices[key].to_csv(path + filename + '_{}.csv'.format(key), sep=',', header=val, index_label=False)
    return matrices

def convolve_HRF(model_df, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12):
    
    """
    Convolve each column of matrix with a HRF 
    
    Inputs:
    - model_df : dataframe of full group model
    - tr : sampling interval in seconds (fMRI TR)
    - hrf_p : parameter of HRF
    - hrf_q : parameter of HRF
    - dur : duration of HRF, in seconds

    Outputs:
    - group_convolved : dataframe of full group model convolved
    """

    # Define HRF
    hrf_t = np.arange(0, dur+0.5, tr)  # A typical HRF lasts 12 secs
    hrf = (hrf_t / (hrf_p * hrf_q)) ** hrf_p * np.exp(hrf_p - hrf_t / hrf_q)

    # Initialize matrix to save result
    group_convolved = np.full(model_df.shape, np.nan)

    # Iterate over columns
    for column in range(model_df.shape[1]):
        model = model_df.iloc[:,column].to_numpy()
        model_conv = np.convolve(model, hrf, mode='full')[:model.shape[0]] # cut last part 
        model_conv = model_conv / np.max(model_conv)
        group_convolved[:,column] = model_conv
    group_convolved = pd.DataFrame(group_convolved, columns=model_df.columns)
    
    return group_convolved
    
    
if __name__ == '__main__':

    # Load group model
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/tagging_carica101_'
    out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/Domains/tagging_carica101_'
    
    model_df = pd.read_csv(data_path + 'group_2su3_df_reduced.csv') 

    # Convolve model
    model_conv = convolve_HRF(model_df)

    # Define domains
    domains = {}
    domains['model3'] = {
    'space_movement': ['context_0', 'context_1', 'context_2', 'inter_scale', 'dinamicity', 'eff_visibility',
                        'main_effector_0', 'main_effector_1', 'main_effector_2', 'main_effector_3', 
                        'main_effector_4', 'main_effector_5', 'main_effector_6', 'main_effector_7', 
                        'main_effector_8', 'main_effector_9', 'main_effector_10'],
    'agent_objective': ['target_0','target_1', 'target_2', 'target_3', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],
    'social_connectivity': ['sociality', 'touch_1', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
    'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
    'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'predictability']}

    # Create domain matrices and save csv
    get_domain_mat(model_conv, domains['model3'], out_path, 'group_2su3_convolved')