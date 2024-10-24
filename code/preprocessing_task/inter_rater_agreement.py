import numpy as np
import pandas as pd
from utils.similarity_measures import canonical_correlation, linear_cka
import matplotlib.pyplot as plt

def plot_corrs(results, domains, measure, path):
   
    # Parameters
    n_rater_pairs = results.shape[0]
    n_domains = results.shape[1]
    rater_pairs = ['AI vs LM', 'AI vs LT', 'LM vs LT']
    bar_width = 0.2

    x = np.arange(n_domains)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i in range(n_rater_pairs):
        ax.bar(x + i * bar_width, results[i], bar_width, label=rater_pairs[i])

    # Labels and layout
    ax.set_xlabel('Domains')
    ax.set_ylabel('{}'.format(measure))
    ax.set_title('{} between raters for each domain'.format(measure))
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(domains, rotation=45)
    ax.set_ylim(0,1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Legend
    ax.legend(title='Raters')

    # Save
    plt.savefig(path + 'inter-rater_{}_notOHE.png'.format(measure))

if __name__ == '__main__':
    
    taggers = ['LM', 'AI', 'LT']
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'

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
    AI = pd.read_csv(path + '{}_ds_notOHE.csv'.format('AI'), sep=',')
    LM = pd.read_csv(path + '{}_ds_notOHE.csv'.format('LM'), sep=',')
    LT = pd.read_csv(path + '{}_ds_notOHE.csv'.format('LT'), sep=',')
    
    # Init results
    CKA_res = np.zeros((3, len(domains)))
    dbCKA_res = np.zeros((3, len(domains)))
    CCA_res = np.zeros((3, len(domains)))

    for d, dom in enumerate(domains.keys()):
        ai = AI[domains_notOHE[dom]].to_numpy()
        lm = LM[domains_notOHE[dom]].to_numpy()
        lt = LT[domains_notOHE[dom]].to_numpy()

        CKA_res[0, d] = linear_cka(ai, lm)
        CKA_res[1, d] = linear_cka(ai, lt)
        CKA_res[2, d] = linear_cka(lm, lt)
        
        dbCKA_res[0, d] = linear_cka(ai, lm, debiasing=True)
        dbCKA_res[1, d] = linear_cka(ai, lt, debiasing=True)
        dbCKA_res[2, d] = linear_cka(lm, lt, debiasing=True)
        
        CCA_res[0, d] = canonical_correlation(ai, lm)[0]
        CCA_res[1, d] = canonical_correlation(ai, lt)[0]
        CCA_res[2, d] = canonical_correlation(lm, lt)[0]
    
    plot_corrs(CKA_res, list(domains.keys()), 'CKA', path)
    plot_corrs(dbCKA_res, list(domains.keys()), 'debiasedCKA', path)
    plot_corrs(CCA_res, list(domains.keys()), 'CCA', path)
    

    print('d')