import pandas as pd
import numpy as np

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
    
    return df_downsampled


def time_to_dseconds(time_str):

    """
    Convert onset and offset times to decimals of seconds

    Inputs:
    - time_str : string of type mm:ss.dd

    Output:
    - int : time in decimal of seconds
    """

    minutes, seconds = map(float, time_str.split(':'))
    return int((minutes * 60 + seconds)*10)


if __name__ == '__main__':

    # Set to display all columns
    pd.set_option('display.max_columns', None)

    # Load csv file
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_PreprocessedTagging/tagging_carica101_'
    df_all = pd.read_csv(data_path + 'combined_preprocessed.csv', sep=',')

    # Specify run durations (for later)
    durations = {'r1':'08:56.4', 'r2':'07:30.7', 'r3':'10:40.7', 'r4':'10:50.2', 'r5':'07:52.6', 'r6':'08:00.3'}


    # Convert onset and offset columns to decimals of seconds
    df_all['onset_ds'] = df_all.apply(lambda x: (float(x.onset.split(':')[0])*60 + float(x.onset.split(':')[1]))*10, axis=1).astype('int64')
    df_all['offset_ds'] = df_all.apply(lambda x: (float(x.offset.split(':')[0])*60 + float(x.offset.split(':')[1]))*10, axis=1).astype('int64')

    # Select columns of interest and columns to drop
    coi_binary = ['inter_scale', 'predictability', 'agent_H_NH', 'eff_visibility', 'sociality', 'dinamicity', 'durativity', 'telicity', 'iterativity', 'transitivity', 
                'tool_mediated','faces', 'ToM',  'EBL', 'EIA', 'gesticolare', 'simbolic_gestures', 'people_present']

    coi_ohe = ['context', 'main_effector', 'touch', 'multi_ag_vs_jointact', 'target']

    ctd = ['Unnamed: 0', 'Unnamed: 1', 'act', 'act_label', 'obj_label'] #'onset', 'offset'

    features = ['inter_scale', 'predictability', 'agent_H_NH', 'eff_visibility', 'sociality', 'dinamicity',
                'durativity', 'telicity', 'iterativity', 'transitivity', 'tool_mediated', 'context', 
                'main_effector', 'multi_ag_vs_jointact', 'target', 'faces', 'ToM', 'touch', 'EBL', 'EIA',
                'gesticolare', 'simbolic_gestures', 'people_present']

    # Add column of ones to df to indicate that an action is present here
    df_all['action_present'] = 1

    # Drop useless columns
    df_all.drop(ctd, axis=1, inplace=True)

    # Onehot econding
    df_all_ohe = pd.get_dummies(data=df_all, columns=coi_ohe, dummy_na=False, dtype=int) #columns without NaNs as possible values
    #df_all_ohe = pd.get_dummies(data=df_all_ohe, columns=coi_nan, dummy_na=True) #columns with NaNs

    # Define features of interest for final matrix
    features_inds = [idx for idx, val in enumerate(df_all_ohe.columns.tolist()) if val not in ['onset', 'offset', 'run', 'rater', 'onset_ds', 'offset_ds']]
    features = [val for idx, val in enumerate(df_all_ohe.columns.tolist()) if val not in ['onset', 'offset', 'run', 'rater', 'onset_ds', 'offset_ds']]

    # Iterate over raters
    raters = ['LM', 'AI', 'LT']

    # Initialize dictionary to save matrices
    expanded_dfs = {}
    run_cuts = np.array([536,450,640,650,472,480])*10

    for rat_n, rater in enumerate(raters):
        df_r = df_all_ohe[df_all_ohe.rater == rater]
        
        # Initialize a matrix for each rater
        df_exp_r = np.zeros((1, len(features)))
        
        # Iterate over runs
        for r,run in enumerate([*range(1,7)]):
            df_r_run = df_r[df_r.run == run]
            
            # Retrieve run duration in decimals of seconds
            #duration = time_to_dseconds(durations['r{}'.format(run)])
            duration = run_cuts[r]
            
            # Create a list of timepoints from 0 to duration with the desired resolution
            timepoints = np.arange(0, duration)
            
            # Initialize matrix of zeros of shape = n_timepoints by n_columns ohe (excluding onset and offset)
            df_exp = np.zeros((duration, len(features)))
            
            # Iterate over timepoints
            for time in timepoints:
                mask = (df_r_run['onset_ds'] <= time) & (df_r_run['offset_ds'] >= time)
                if mask.any(): # Check if timepoint is included in the interval of at least one action
                    event_features = df_r_run[mask].iloc[:, features_inds]  # Select features excluding time-related columns, rater and run
                    event_features = event_features.sum().to_numpy() # If more than one action for timepoint, sum them up and transform to np.array
                    event_features[event_features > 0] = 1 # Set all positive values back to 1
                    df_exp[time] = event_features # Assign array to matrix at index = timepoint
                else: # If timepoint not in any action just assign 0 to all features (change to -1?)
                    df_exp[time] = np.zeros(len(features))
            
            # Concatenate matrix to other runs
            df_exp_r = np.concatenate((df_exp_r, df_exp), axis=0)
            
        # Delete first row of each rater matrix (used to initialize)
        df_exp_r = df_exp_r[1:]
        
        # Save matrix in dictionary
        expanded_dfs[rater] = df_exp_r


    # Save files
    lm = pd.DataFrame(expanded_dfs['LM'], columns=features)
    ai = pd.DataFrame(expanded_dfs['AI'], columns=features)
    lt = pd.DataFrame(expanded_dfs['LT'], columns=features)

    lm.to_csv(data_path + 'LM_expanded.csv', sep=',', index_label=False)
    ai.to_csv(data_path + 'AI_expanded.csv', sep=',', index_label=False)
    lt.to_csv(data_path + 'LT_expanded.csv', sep=',', index_label=False)


    # Downsample to resolution 2sec
    # Load csv file
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_PreprocessedTagging/tagging_carica101_'
    lm = pd.read_csv(data_path + 'LM_expanded.csv', sep=',')#drop(columns='Unnamed: 0')
    ai = pd.read_csv(data_path + 'AI_expanded.csv', sep=',')#.drop(columns='Unnamed: 0')
    lt = pd.read_csv(data_path + 'LT_expanded.csv', sep=',')#.drop(columns='Unnamed: 0')

    lm_ds_bin = downsample(lm, 2, 'binary')
    lm_ds_norm = downsample(lm, 2, 'normalized')

    ai_ds_bin = downsample(ai, 2, 'binary')
    ai_ds_norm = downsample(ai, 2, 'normalized')

    lt_ds_bin = downsample(lt, 2, 'binary')
    lt_ds_norm = downsample(lt, 2, 'normalized')

    # Save files
    np.savez(data_path + 'all_down_sampled_binary.npz', lm_ds=lm_ds_bin, ai_ds=ai_ds_bin, lt_ds=lt_ds_bin, features=features)
    np.savez(data_path + 'all_down_sampled_normalized.npz', lm_ds=lm_ds_norm, ai_ds=ai_ds_norm, lt_ds=lt_ds_norm, features=features)