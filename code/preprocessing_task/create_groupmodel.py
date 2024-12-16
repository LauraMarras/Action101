import pandas as pd
import numpy as np


def expand_mat(df, duration, features_inds):
    
    """
    Expand dataframe on a specific time resolution 
    
    Inputs:
    - df : dataframe of shape n_actions by features
    - duration : total number of t_points (define time resolution)
    - indices of features to copy in expanded matrix

    Outputs:
    - expanded_df : matrix of shape n_tpoints by features
    """

    # Initialize matrix of zeros of shape = n_timepoints by n_columns ohe (excluding onset and offset)
    df_exp = np.zeros((duration, len(features_inds)))
    
    # Iterate over timepoints
    for time in range(duration):
        mask = (df['onset_ds'] <= time) & (df['offset_ds'] >= time)
        if mask.any(): # Check if timepoint is included in the interval of at least one action
            event_features = df[mask].iloc[:, features_inds]  # Select features excluding time-related columns, rater and run
            event_features = event_features.sum().to_numpy() # If more than one action for timepoint, sum them up and transform to np.array
            event_features[event_features > 0] = 1 # Set all positive values back to 1
            df_exp[time] = event_features # Assign array to matrix at index = timepoint
        else: # If timepoint not in any action just assign 0 to all features
            df_exp[time] = np.zeros(len(features_inds))

    return df_exp

def expand_mat_allruns(df, features_inds, run_cuts, ratername, path, n_runs=6):
    
    """
    Calls expand_mat for each run and concatenates all expanded runs together
    
    Inputs:
    - df : dataframe of shape n_actions by features for multiple runs of single rater
    - indices of features to copy in expanded matrix    
    - run_cuts : total number of t_points (define time resolution)
    - n_runs : number of runs; default = 6

    Outputs:
    - df_exp_r : dataframe of shape n_tpoints by features, of all runs concatenated
    """

    # Initialize a matrix for each rater
    df_exp_r = np.zeros((1, len(features_inds)))
    
    # Iterate over runs
    for r,run in enumerate([*range(1,n_runs+1)]):
        df_r_run = df[df.run == run]
        
        ## Expand to time
        df_exp = expand_mat(df_r_run, run_cuts[r], features_inds)
        
        ## Concatenate matrix to other runs
        df_exp_r = np.concatenate((df_exp_r, df_exp), axis=0)


    ## Convert to dataframe
    features = [feat for idx, feat in enumerate(df_r_run.columns.tolist()) if idx in features_inds]
    df_exp_r = pd.DataFrame(df_exp_r[1:], columns=features)
    
    # Save
    if path:
        df_exp_r.to_csv(path + '{}_expanded.csv'.format(ratername), sep=',', index_label=False)
    
    return df_exp_r

def downsample(df, resolution=2, method='binary'):
    
    """
    Downsample dataframe to specific resolution in seconds 
    
    Inputs:
    - df : dataframe of timpoints by columns
    - resolution : sampling interval in seconds; default = 2 (fMRI TR)
    - method : 'binary' or 'normalized', whether to keep binary or to 'normalize'; default = 'binary' 
    Outputs:
    - df_downsampled : dataframe downsampled of shape n_timepoints (in new resolution) by columns
    """

    # Define arrays of cut ons and cut offs
    cuts = np.arange(0, df.shape[0]+resolution*10, resolution*10)
    cut_on = cuts[:-1]
    cut_off = cuts[1:]
    cut_off[-1] = df.shape[0]

    # Initialize final matrix
    df_downsampled = np.zeros((len(cut_on), df.shape[1]))

    # Iterate over cut ons
    for c, c_on in enumerate(cut_on):
        c_off = cut_off[c]

        summed_rows = df.loc[c_on:c_off, :].sum(axis=0)    
        if method == 'binary':
            summed_rows[summed_rows > 0] = 1
            df_downsampled[c,:] = summed_rows
        
        elif method == 'normalized':
            df_downsampled[c,:] = np.divide(summed_rows, np.ones(df.shape[1])*20)
    
    df_downsampled = pd.DataFrame(df_downsampled, columns=df.columns)
    return df_downsampled

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
    hrf_t = np.arange(0, dur+0.005, tr)  # A typical HRF lasts 12 secs
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
    
    # Define domains
    upsample = False

    domains = {}
    domains['conceptual_1'] = {
    'space_movement': ['context_0', 'context_1', 'context_2', 'inter_scale',  'eff_visibility',
                        'main_effector_0', 'main_effector_1', 'main_effector_2', 'main_effector_3', 
                        'main_effector_4', 'main_effector_5', 'main_effector_6', 'main_effector_7', 
                        'main_effector_8', 'main_effector_9', 'main_effector_10'],
    'agent_objective': ['target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],
    'social_connectivity': ['sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
    'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
    'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'dinamicity']}
    
    domains['conceptual_2'] = {
    'space': ['context_0', 'context_1', 'context_2', 'inter_scale'],
    'movement': ['eff_visibility', 'main_effector_0', 'main_effector_1', 'main_effector_2', 
                 'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6',
                  'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10'],
    'agent_objective': ['target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],
    'social_connectivity': ['sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
    'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
    'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'dinamicity']}

    # Set to display all columns
    pd.set_option('display.max_columns', None)

    # Load csv files
    data_path = '/home/laura.marras/Documents/Repositories/Action101/data/preprocessed_tagging/tagging_carica101_'
    df_all = pd.read_csv(data_path + 'all_preprocessed.csv', sep=',')

    # Select columns to OHE and to drop
    ctd = ['act', 'act_label', 'obj_label', 'onset', 'offset']
    columns_ohe = ['context', 'main_effector', 'touch', 'multi_ag_vs_jointact', 'target']

    # Onehot econding
    df_all_ohe = pd.get_dummies(data=df_all.drop(ctd, axis=1), columns=columns_ohe, dummy_na=False, dtype=int) #columns without NaNs as possible values
    
    # Expand matrix to time resolution 0.1 sec
    ## Select columns to keep in final matrix
    features_inds = [idx for idx, val in enumerate(df_all_ohe.columns.tolist()) if val not in ['run', 'onset_ds', 'offset_ds', 'rater']]
    features = [feat for feat in df_all_ohe.columns.tolist() if feat not in ['run', 'onset_ds', 'offset_ds', 'rater']]
    
    ## Specify duration of each run in fMRI
    run_cuts = np.array([536,450,640,650,472,480])*10
    
    df_exp_LM = expand_mat_allruns(df_all_ohe[df_all_ohe.rater == 'LM'], features_inds, run_cuts, 'LM', data_path, 6)
    df_exp_AI = expand_mat_allruns(df_all_ohe[df_all_ohe.rater == 'AI'], features_inds, run_cuts, 'AI', data_path, 6)
    df_exp_LT = expand_mat_allruns(df_all_ohe[df_all_ohe.rater == 'LT'], features_inds, run_cuts, 'LT', data_path, 6)
    
    # Create group models
    group_model = df_exp_LM+df_exp_AI+df_exp_LT # Model that keeps 1 if 2 out of 3 raters had 1, less conservative
    group_model[group_model<2] = 0
    group_model[group_model>1] = 1

    # Upsample to 0.05sec time resolution
    if upsample:
        group_upsampled = group_model
        group_upsampled['timedelta'] = pd.TimedeltaIndex(np.arange(group_model.shape[0])/10, unit='S')
        group_upsampled = group_upsampled.set_index('timedelta')
        group_upsampled = group_upsampled.resample('0.05S').interpolate(method='nearest')
        group_upsampled.reset_index(drop=True, inplace=True)
        group_upsampled.loc[len(group_upsampled)] = 0
        group_model.drop('timedelta', axis=1, inplace=True)

    # Downsample to resolution 2sec
    group_downsampled = downsample(group_model, 2, 'binary')
    df_AI_ds = downsample(df_exp_AI, 2, 'binary')
    df_LM_ds = downsample(df_exp_LM, 2, 'binary')
    df_LT_ds = downsample(df_exp_LT, 2, 'binary')

    # Convolve with HRF
    group_conv_ds = convolve_HRF(group_downsampled, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12)
    df_AI_conv_ds = convolve_HRF(df_AI_ds, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12)
    df_LM_conv_ds = convolve_HRF(df_LM_ds, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12)
    df_LT_conv_ds = convolve_HRF(df_LT_ds, tr=2, hrf_p=8.6, hrf_q=0.547, dur=12)

    # Create domain matrices and save csv
    out_path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'
    if upsample:
        domains_us_bin = get_domain_mat(group_upsampled, domains['conceptual_2'], out_path+'domains/', 'group_us_binary')
        group_upsampled.to_csv(out_path + 'group_bin_us.csv', sep=',', index_label=False)
    domains_ds_bin = get_domain_mat(group_downsampled, domains['conceptual_2'], out_path+'domains/', 'group_ds_binary')
    domains_ds_conv = get_domain_mat(group_conv_ds, domains['conceptual_2'], out_path+'domains/', 'group_ds_conv')

    # Save all dframes
    group_conv_ds.to_csv(out_path + 'group_conv_ds.csv', sep=',', index_label=False)
    group_downsampled.to_csv(out_path + 'group_bin_ds.csv', sep=',', index_label=False)

    df_AI_ds.to_csv(out_path + 'AI_bin_ds.csv', sep=',', index_label=False)
    df_LM_ds.to_csv(out_path + 'LM_bin_ds.csv', sep=',', index_label=False)
    df_LT_ds.to_csv(out_path + 'LT_bin_ds.csv', sep=',', index_label=False)
    df_AI_conv_ds.to_csv(out_path + 'AI_conv_ds.csv', sep=',', index_label=False)
    df_LM_conv_ds.to_csv(out_path + 'LM_conv_ds.csv', sep=',', index_label=False)
    df_LT_conv_ds.to_csv(out_path + 'LT_conv_ds.csv', sep=',', index_label=False)

    print('d')