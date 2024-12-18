import numpy as np
import pandas as pd
from utils.similarity_measures import linear_cka

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

    models = ['motion', 'gpt4', 'power', 'relu6', 'relu5-1'] #'relu3-1'

    # Load CKA results
    try:
        CKA_df = pd.read_csv('{}comp_models/CKA{}_doms{}_compmodels.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'), index_col=0)

    except FileNotFoundError:
        
        # Load group model
        model_group = pd.read_csv('{}group_{}_ds.csv'.format(path, 'conv' if convolved else 'bin'), sep=',')

        # Initialize results matrix
        CKA_res = np.empty((len(models), len(domains)))

        for m, model in enumerate(models):
            try:
                X = np.loadtxt('{}comp_models/{}_task-movie_allruns.tsv'.format(path, model))
            except:
                X = np.loadtxt('{}comp_models/{}_task-movie_allruns.tsv.gz'.format(path, model))
        
            for d, dom in enumerate(domains.keys()):
                Y = model_group[domains[dom]].to_numpy()
                CKA_res[m,d] = linear_cka(X, Y, debiasing=debiasing)

        # Save results
        CKA_df = pd.DataFrame(CKA_res, columns=domains.keys(), index=models)
        CKA_df.to_csv('{}comp_models/CKA{}_doms{}_compmodels.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'))