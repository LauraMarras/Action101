import numpy as np
import pandas as pd
from utils.similarity_measures import canonical_correlation, linear_cka
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    
    # Load domains
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    full_model = {'full_model': np.hstack([domains[d] for d in domains_list])}
    full = full_model['full_model']
    domains['full'] = full

    # Load computational models
    models_path = '/home/laura.marras/Documents/Repositories/Action101/data/models/comp_models/'
    models = ['motion', 'gpt4', 'power', 'relu6', 'relu5-1'] #'relu3-1'

    # Initialize res matrices
    cka_res = np.empty((len(models), len(domains)))
    dbcka_res = np.empty((len(models), len(domains)))

    for m, model in enumerate(models):
        try:
            model_data = np.loadtxt('{}{}_task-movie_allruns.tsv'.format(models_path, model))
        except:
            model_data = np.loadtxt('{}{}_task-movie_allruns.tsv.gz'.format(models_path, model))
    
        for d, dom in enumerate(domains.keys()):
            cka_res[m,d] = linear_cka(model_data, domains[dom], debiasing=False)
            dbcka_res[m,d] = linear_cka(model_data, domains[dom], debiasing=True)

    print('d')