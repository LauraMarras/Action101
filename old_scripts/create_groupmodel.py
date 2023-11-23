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
'space_movement': ['context_0', 'context_1', 'context_2', 'inter_scale', 'eff_visibility', #'dinamicity',
                    'main_effector_0', 'main_effector_1', 'main_effector_2', 'main_effector_3', 'main_effector_4', 'main_effector_5', 
                    'main_effector_6', 'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10'],
'agent_objective': ['target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'], #'target_2', 'target_3', 
'social_connectivity': ['target_2', 'target_3', 'sociality', 'touch_1', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present'],
'emotion_expression': ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],
'linguistic_predictiveness': ['durativity', 'telicity', 'iterativity', 'dinamicity']}} #'predictability'

# Re-order features
order = []
for key, val in domains['features'].items():
    order += [features.index(c) for c in val]
features_reorder = [features[o] for o in order]

group_df_reduced = group_df_reduced.reindex(columns=features_reorder)

# Save group model to csv file
group_df.to_csv(out_path + 'group_2su3_df.csv', sep=',')
group_df_reduced.to_csv(out_path + 'group_2su3_df_reduced.csv', sep=',')