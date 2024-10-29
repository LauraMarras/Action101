import numpy as np
import pandas as pd
from utils.similarity_measures import canonical_correlation, linear_cka, cka, gram_linear, feature_space_linear_cka
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

    try:
        # Load the results
        dbcka_df = pd.read_csv(models_path + 'cka_domains.csv', index_col=0)

    except FileNotFoundError:
        # Initialize res matrices
        dbcka_res = np.empty((len(models), len(domains)))

        for m, model in enumerate(models):
            try:
                X = np.loadtxt('{}{}_task-movie_allruns.tsv'.format(models_path, model))
            except:
                X = np.loadtxt('{}{}_task-movie_allruns.tsv.gz'.format(models_path, model))
        
            for d, dom in enumerate(domains.keys()):
                Y = domains[dom]
                dbcka_res[m,d] = linear_cka(X, Y, debiasing=True)

        # Save results
        dbcka_df = pd.DataFrame(dbcka_res, columns=domains.keys(), index=models)
        dbcka_df.to_csv(models_path + 'cka_domains.csv')

    # Plot
    dbcka_melted = dbcka_df.reset_index().melt(id_vars='index', var_name='Domain', value_name='Similarity')
    dbcka_melted.rename(columns={'index': 'Model'}, inplace=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=dbcka_melted, x='Domain', y='Similarity', hue='Model', palette='tab10')

    # Set title, labels, axes and legend
    plt.title('CKA similarity between computational models and domains')
    plt.xlabel('Domain')
    plt.ylabel('CKA debiased')
    plt.ylim(0,0.4)
    plt.legend(title='Model', loc='upper right', frameon=False)  #bbox_to_anchor=(1.05, 1),
    plt.xticks(rotation=45)
    plt.tight_layout()

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save
    plt.savefig(models_path + 'cka.png')

    # Set up the heatmap plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(dbcka_df, annot=True, fmt=".2f", cmap='rocket_r', cbar_kws={'label': 'CKA Similarity'})

    # Customize the plot
    plt.title('CKA Similarity between Computational Models and Domains')
    plt.xlabel('Domain')
    plt.ylabel('Model')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()

    # Save the plot
    plt.savefig(models_path + 'cka_heatmap.png')