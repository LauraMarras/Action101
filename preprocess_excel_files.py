import pandas as pd
import numpy as np

# Set to display all columns
pd.set_option('display.max_columns', None)

# Load Excel files
data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_RawTagging/tagging_carica101_'
out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_PreprocessedTagging/tagging_carica101_'

## Laura
ex_laura = pd.read_excel(data_path + 'Laura.xlsx', sheet_name=None)
df_laura = pd.DataFrame()
for sheet in ex_laura: # Extract each excel sheet and concatenate to main dataframe
    df_laura = pd.concat([df_laura, ex_laura[sheet]], axis=0)

## Alessandro    
ex_ale = pd.read_excel(data_path + 'Alessandro.xlsx', sheet_name=None)
df_ale = pd.DataFrame()
for sheet in ex_ale: # Extract each excel sheet and concatenate to main dataframe
    df_ale = pd.concat([df_ale, ex_ale[sheet]], axis=0)

## Lorenzo
df_lore = pd.read_excel(data_path + 'Lorenzo_new.xlsx')

# Clean dataframes
df_laura['act'] = df_laura.apply(lambda x: x['act.1'] if x['run'] == 3 else x['act'], axis=1)
df_laura.drop(['Audio', 'Video', 'Unnamed: 0', 'action', 'act.1'], axis=1, inplace=True)

df_laura = df_laura[pd.to_numeric(df_laura['main_effector'], errors='coerce').notnull()]
df_laura['main_effector'] = df_laura['main_effector'].astype('int64')

df_laura.loc[df_laura.touch==-1, 'touch'] = 2

df_laura.loc[df_laura.act == 'act_667', 'eff_visibility'] = 0
df_laura.loc[df_laura.act == 'act_765', 'sociality'] = 0

df_ale.loc[df_ale['Unnamed: 0'] == 'act_1009', 'sociality'] = 1
df_lore.loc[df_lore['Unnamed: 0'] == 'act_85', 'People_present'] = 0

df_ale.people_present = df_ale.people_present.fillna(0).astype('int64')
df_ale.multi_ag_vs_jointact = df_ale.multi_ag_vs_jointact.apply(lambda x: np.nan if x=='naN' else float(x))

df_laura.multi_ag_vs_jointact = df_laura.multi_ag_vs_jointact + 1
df_laura.multi_ag_vs_jointact = df_laura.multi_ag_vs_jointact.fillna(0).astype('int64')

df_ale.multi_ag_vs_jointact = df_ale.multi_ag_vs_jointact + 1
df_ale.multi_ag_vs_jointact = df_ale.multi_ag_vs_jointact.fillna(0).astype('int64')

df_lore.multi_ag_vs_jointact = df_lore.multi_ag_vs_jointact + 1
df_lore.multi_ag_vs_jointact = df_lore.multi_ag_vs_jointact.fillna(0).astype('int64')

df_lore.agent_H_NH = df_lore.agent_H_NH.apply(lambda x: abs(x-1))

df_ale.rename(columns={'Unnamed: 0':'act'}, inplace=True)
df_lore.rename(columns={'Unnamed: 0':'act', 'complexity':'predictability', 'Gesticolare':'gesticolare', 'Symbolic Gestures':'simbolic_gestures', 'People_present':'people_present'}, inplace=True)

df_lore.target = df_lore.target.fillna(4).astype('int64')

df_laura = df_laura[df_ale.columns.tolist()]
df_lore = df_lore.reindex(columns=df_ale.columns.tolist(), fill_value=None)

# Change coding of columns who get very low sparsity?
# df_lore.dinamicity = df_lore.dinamicity.apply(lambda x: abs(x-1))
# df_laura.dinamicity = df_laura.dinamicity.apply(lambda x: abs(x-1))
# df_ale.dinamicity = df_ale.dinamicity.apply(lambda x: abs(x-1))

# df_lore.iterativity = df_lore.iterativity.apply(lambda x: abs(x-1))
# df_laura.iterativity = df_laura.iterativity.apply(lambda x: abs(x-1))
# df_ale.iterativity = df_ale.iterativity.apply(lambda x: abs(x-1))

# df_lore.eff_visibility = df_lore.eff_visibility.apply(lambda x: abs(x-1))
# df_laura.eff_visibility = df_laura.eff_visibility.apply(lambda x: abs(x-1))
# df_ale.eff_visibility = df_ale.eff_visibility.apply(lambda x: abs(x-1))


# Add column indicating rater to each dataframe
df_laura['rater'] = 'LM'
df_ale['rater'] = 'AI'
df_lore['rater'] = 'LT'

# Concatenate dataframes in a unique one
df_all = pd.concat([df_laura, df_ale, df_lore], axis=0, keys=['R1', 'R2', 'R3'])

# Save files
df_laura.to_csv(out_path + 'LM_preprocessed.csv', sep=',')
df_ale.to_csv(out_path + 'AI_preprocessed.csv', sep=',')
df_lore.to_csv(out_path + 'LT_preprocessed.csv', sep=',')

df_all.to_csv(out_path + 'combined_preprocessed.csv', sep=',')