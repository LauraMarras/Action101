import pandas as pd
import numpy as np
import re

def split_string(x, col):
    m, s, d = re.split('[:|.]', x[col])
    ds = float(m)*600 + float(s)*10 + float(d[0])
    return int(ds)

if __name__ == '__main__':

    # Load Excel files
    data_path = 'Action101/data/raw_tagging/tagging_carica101_'
    out_path = 'Action101/data/preprocessed_tagging/tagging_carica101_'
 
    ## Laura
    ex_LM = pd.read_excel(data_path + 'LM.xlsx', sheet_name=None)
    df_LM = pd.DataFrame()
    for sheet in ex_LM: # Extract each excel sheet and concatenate to main dataframe
        df_LM = pd.concat([df_LM, ex_LM[sheet]], axis=0)

    ## Alessandro    
    ex_AI = pd.read_excel(data_path + 'AI.xlsx', sheet_name=None)
    df_AI = pd.DataFrame()
    for sheet in ex_AI: # Extract each excel sheet and concatenate to main dataframe
        df_AI = pd.concat([df_AI, ex_AI[sheet]], axis=0)

    ## Lorenzo
    df_LT = pd.read_excel(data_path + 'LT.xlsx')

    # Clean dataframes
    ## Alessandro
    df_AI.loc[df_AI['Unnamed: 0'] == 'act_1009', 'sociality'] = 1
    df_AI.loc[df_AI['Unnamed: 0'] == 'act_787', 'offset'] = '05:09:5'
    df_AI.people_present = df_AI.people_present.fillna(0).astype('int64')
    df_AI.multi_ag_vs_jointact = df_AI.multi_ag_vs_jointact.apply(lambda x: np.nan if x=='naN' else float(x))
    df_AI.multi_ag_vs_jointact = df_AI.multi_ag_vs_jointact + 1
    df_AI.multi_ag_vs_jointact = df_AI.multi_ag_vs_jointact.fillna(0).astype('int64')
    df_AI.rename(columns={'Unnamed: 0':'act'}, inplace=True)
    df_AI['onset_ds'] = df_AI.apply(lambda x: split_string(x, 'onset'), axis=1) #.apply(lambda x: (float(x.onset.split(':')[0])*600 + float(x.onset.split(':')[1])*10 + float(x.onset.split(':')[2])), axis=1).astype('int64')
    df_AI['offset_ds'] = df_AI.apply(lambda x: split_string(x, 'offset'), axis=1) #.apply(lambda x: (float(x.offset.split(':')[0])*600 + float(x.offset.split(':')[1])*10 + float(x.offset.split(':')[2])), axis=1).astype('int64')
    df_AI['rater'] = 'AI'
    df_AI['action_present'] = 1
    df_AI['duration_seconds'] = (df_AI.offset_ds - df_AI.onset_ds)/10
    df_AI = df_AI[df_AI.duration_seconds > 0]

    ## Laura
    df_LM['act'] = df_LM.apply(lambda x: x['act.1'] if x['run'] == 3 else x['act'], axis=1)
    df_LM.drop(['Audio', 'Video', 'Unnamed: 0', 'action', 'act.1'], axis=1, inplace=True)
    df_LM = df_LM[pd.to_numeric(df_LM['main_effector'], errors='coerce').notnull()]
    df_LM['main_effector'] = df_LM['main_effector'].astype('int64')
    df_LM.loc[df_LM.touch==-1, 'touch'] = 2
    df_LM.loc[df_LM.act == 'act_667', 'eff_visibility'] = 0
    df_LM.loc[df_LM.act == 'act_765', 'sociality'] = 0
    df_LM.multi_ag_vs_jointact = df_LM.multi_ag_vs_jointact + 1
    df_LM.multi_ag_vs_jointact = df_LM.multi_ag_vs_jointact.fillna(0).astype('int64')
    df_LM['onset_ds'] = df_LM.apply(lambda x: split_string(x, 'onset'), axis=1) #.apply(lambda x: (float(x.onset.split(':')[0])*60 + float(x.onset.split(':')[1]))*10, axis=1).astype('int64')
    df_LM['offset_ds'] = df_LM.apply(lambda x: split_string(x, 'offset'), axis=1) #.apply(lambda x: (float(x.offset.split(':')[0])*60 + float(x.offset.split(':')[1]))*10, axis=1).astype('int64')
    df_LM['rater'] = 'LM'
    df_LM['action_present'] = 1
    df_LM['duration_seconds'] = (df_LM.offset_ds - df_LM.onset_ds)/10
    df_LM = df_LM[df_AI.columns.tolist()]
    df_LM = df_LM[df_LM.duration_seconds > 0]

    ## Lorenzo
    df_LT.loc[df_LT['Unnamed: 0'] == 'act_85', 'People_present'] = 0
    df_LT.multi_ag_vs_jointact = df_LT.multi_ag_vs_jointact + 1
    df_LT.multi_ag_vs_jointact = df_LT.multi_ag_vs_jointact.fillna(0).astype('int64')
    df_LT.agent_H_NH = df_LT.agent_H_NH.apply(lambda x: abs(x-1))
    df_LT.rename(columns={'Unnamed: 0':'act', 'complexity':'predictability', 'Gesticolare':'gesticolare', 'Symbolic Gestures':'simbolic_gestures', 'People_present':'people_present'}, inplace=True)
    df_LT.target = df_LT.target.fillna(4).astype('int64')
    df_LT['onset_ds'] = df_LT.apply(lambda x: split_string(x, 'onset'), axis=1) #.apply(lambda x: (float(x.onset.split(':')[0])*60 + float(x.onset.split(':')[1]))*10, axis=1).astype('int64')
    df_LT['offset_ds'] = df_LT.apply(lambda x: split_string(x, 'offset'), axis=1) #.apply(lambda x: (float(x.offset.split(':')[0])*60 + float(x.offset.split(':')[1]))*10, axis=1).astype('int64')
    df_LT['rater'] = 'LT'
    df_LT['action_present'] = 1
    df_LT['duration_seconds'] = (df_LT.offset_ds - df_LT.onset_ds)/10
    df_LT = df_LT.reindex(columns=df_AI.columns.tolist(), fill_value=None)
    df_LT = df_LT[df_LT.duration_seconds > 0]

    # Concatenate dataframes in a unique one
    df_all = pd.concat([df_LM, df_AI, df_LT], axis=0)

    # Save files
    df_LM.to_csv(out_path + 'LM_preprocessed_new.csv', sep=',', index_label=False)
    df_AI.to_csv(out_path + 'AI_preprocessed_new.csv', sep=',', index_label=False)
    df_LT.to_csv(out_path + 'LT_preprocessed_new.csv', sep=',', index_label=False)

    df_all.to_csv(out_path + 'all_preprocessed_new.csv', sep=',', index_label=False)