import pandas as pd
import numpy as np

# Define downsampling
downsample = 'binary'

# Load files
data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_PreprocessedTagging/tagging_carica101_'
out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/tagging_carica101_'

all_down_sampled = np.load(data_path + 'all_down_sampled_{}.npz'.format(downsample))

lm = all_down_sampled['lm_ds']
ai = all_down_sampled['ai_ds']
lt = all_down_sampled['lt_ds']

# Load features
features = list(all_down_sampled['features'])

# Create group models
# group_model = lm*ai*lt # Model that only keeps 1 if all 3 raters had 1, more conservative
group_model = lm+ai+lt # Model that keeps 1 if 2 out of 3 raters had 1, less conservative
group_model[group_model<2] = 0
group_model[group_model>1] = 1

group_df = pd.DataFrame(group_model, columns=features).astype(int)

# Check features sparsity
group_sparsity = 1 - np.count_nonzero(group_model, 0)/ float(group_model.shape[0])
cols_low_sparsity = list(np.array(features)[group_sparsity>=0.995])

# Drop columns with low sparsity
group_df_reduced = group_df.drop(columns=cols_low_sparsity)
group_df_reduced.drop(columns=['multi_ag_vs_jointact_0', 'touch_0', 'faces', 'action_present'], inplace=True)

# Change coding of columns who get very low sparsity?
# group_df.dinamicity = group_df.dinamicity.apply(lambda x: abs(x-1))
# group_df.durativity = group_df.durativity.apply(lambda x: abs(x-1))
    
# Define domains
domains = { 'features':{
'space_movement': ['context_0', 'context_1', 'context_2', 'inter_scale', 'dinamicity', 'eff_visibility',
                    'main_effector_0', 'main_effector_1', 'main_effector_2', 'main_effector_3', 'main_effector_4', 'main_effector_5', 
                    'main_effector_6', 'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10'],
'agent_objective': ['target_0','target_1', 'target_2', 'target_3', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],
'social_connectivity': ['sociality', 'touch_1', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'predictability']}}

# Re-order features
order = []
for key, val in domains['features'].items():
    order += [features.index(c) for c in val]
features_reorder = [features[o] for o in order]

group_df_reduced = group_df_reduced.reindex(columns=features_reorder)

# Save group model to csv file
group_df.to_csv(out_path + 'group_2su3_df.csv', sep=',')
group_df_reduced.to_csv(out_path + 'group_2su3_df_reduced.csv', sep=',')

# Domain Colors
clist = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'] #['#5390d9', '#5e60ce', '#6930c3', '#e56b6f', '#F19448', '#D4F292']
clist_ext = []
for i, val in enumerate(domains['features'].values()):
    clist_ext += [clist[i] for c in val]

# Define HRF parameters
temporal_resolution = 0.5
hrf_p = 8.6
hrf_q = 0.547
hrf_t = np.arange(0, 12.5, temporal_resolution)  # A typical HRF lasts 12 secs
hrf = (hrf_t / (hrf_p * hrf_q)) ** hrf_p * np.exp(hrf_p - hrf_t / hrf_q)

# Convolve
group_convolved = np.full(group_df_reduced.shape, np.nan)
for column in range(group_df_reduced.shape[1]):
    model = group_df_reduced.iloc[:,column].to_numpy()
    model_conv = np.convolve(model, hrf, mode='full')[:model.shape[0]]
    model_conv = model_conv / np.max(model_conv)
    group_convolved[:,column] = model_conv
group_convolved_df = pd.DataFrame(group_convolved, columns=features_reorder)

# Create domain matrices
domains['matrices'] = {}
domains['matrices_convolved'] = {}
for key, val in domains['features'].items():   
    domain = group_df_reduced[val]
    domain_c = group_convolved_df[val]
    domains['matrices'][key] = domain        
    domains['matrices_convolved'][key] = domain_c        
    domain.to_csv(out_path + 'group_2su3_{}.csv'.format(key), sep=',', header=val, index_label=False)
    domain_c.to_csv(out_path + 'group_2su3_convolved_{}.csv'.format(key), sep=',', header=val, index_label=False)