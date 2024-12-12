import numpy as np
import pandas as pd
from utils.similarity_measures import canonical_correlation, linear_cka
from utils.permutation_schema_func import permutation_schema
import matplotlib.pyplot as plt
import seaborn as sns
from utils.PalColormapImporter import read_pal_file, create_colormap_from_hex

def plot_corrs_subplot(results, domains, measure, path):
   
    # Plot
    fig, axs = plt.subplots(1,2, figsize=(20, 7.5))
    
    for r, result in enumerate(results):
      ax = axs[r]
      sns.heatmap(result, ax=ax, vmin=0, vmax=1, square=True, annot=True, xticklabels=domains, yticklabels=domains, cmap='rocket_r', cbar_kws=dict(pad=0.01,shrink=0.9, label=measure[r]))
    
      # Labels and layout
      ax.set_title('{} between domains'.format(measure[r]))
      ax.set_xticklabels(labels=domains, rotation=45)

    plt.tight_layout()

    # Save
    plt.savefig(path + 'domains_corr_CKA_CCA.png')

def plot_corrs(results, domains, measure, path, pal_file_path=None, width=7.48, height=10.63):

    # Convert afni palette to pyplot cmap
    if pal_file_path:
      hex_colors = read_pal_file(pal_file_path)[1:128] # Select only red to blue part
      hex_colors.reverse()
      cmap = create_colormap_from_hex(hex_colors)
    
    else:
       cmap = 'rocket_r'
    
    # Define Fig size relative to slidesize in inches
    inchsize_size = tuple(np.array([width, height]))
    fig = plt.figure(figsize=inchsize_size, dpi=300)

    # Plot
    sns.heatmap(results, vmin=0, vmax=1, square=True, annot=True,
                  cmap=cmap, cbar_kws=dict(pad=0.01, shrink=0.6, label=measure), 
                  annot_kws={"size":7, "font":'arial'})

    # Labels and layout
    ax = plt.gca()
    ax.set_xticklabels(labels=domains, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(labels=domains, rotation=0)

    plt.tight_layout(pad=0.3)

    # Save
    plt.savefig('{}{}_between_domains.png'.format(path, measure))



def barplot_corrs(results, domains, measures_names, path):
   
    # Parameters
    measures = results.shape[0]
    n_domains = results.shape[1]
    bar_width = 0.2

    x = np.arange(n_domains)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i in range(measures):
        ax.bar(x + i * bar_width, results[i], bar_width, label=measures_names[i])

    # Labels and layout
    ax.set_xlabel('Domains')
    ax.set_ylabel('Similarity')
    ax.set_title('Average similarity between full model and shuffled model for each domain')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(domains, rotation=45)
    ax.set_ylim(0.5,1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Legend
    ax.legend(title='Similarity measures')

    # Save
    plt.savefig(path + 'similarity_fullvsshuffledmodels.png')

def config_plt(textsize=8):

    plt.rc('font', family='Arial')
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=textsize)
    plt.rc('xtick', labelsize=textsize-1)
    plt.rc('ytick', labelsize=textsize-1)

    plt.rc('legend', fontsize=textsize-1)
    plt.rc('legend', loc='best')
    plt.rc('legend', frameon=False)

    plt.rc('grid', linewidth=0.5)
    plt.rc('axes', linewidth=0.5)
    plt.rc('xtick.major', width=0.5)
    plt.rc('ytick.major', width=0.5)

if __name__ == '__main__':
    plot=True
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
                  'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10', 'target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2', 'sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'ToM', 'people_present',
                  'EBL', 'EIA', 'gesticolare', 'simbolic_gestures','durativity', 'telicity', 'iterativity', 'dinamicity']
    }
   
    # Load single tagger tagging
    model_group = pd.read_csv(path + 'group_conv_ds.csv', sep=',')
        
    # Init results
    dbCKA_res = np.zeros((len(domains), len(domains)))
    
    for d1, dom1 in enumerate(domains.keys()):
        domain1 = model_group[domains[dom1]].to_numpy()
        
        for d2, dom2 in enumerate(domains.keys()):
          domain2 = model_group[domains[dom2]].to_numpy()
        
          dbCKA_res[d1, d2] = linear_cka(domain1, domain2, debiasing=False)
        
    # Save
    dbCKA_df = pd.DataFrame(dbCKA_res, columns=domains.keys(), index=domains.keys())
    dbCKA_df.to_csv(path + 'domains/dbCKA_btw_domains.csv')
    
    if plot:
      config_plt()
      domains_labels = ['space', 'effector', 'agent-object', 'social', 'emotion', 'linguistic', 'full']
      plot_path = '/home/laura.marras/Documents/Repositories/Action101/data/plots_final/'
      palette = '/data1/Action_teresi/CCA/code/reds_and_blues.pal'
      
      # Round results to second decimal
      res2plot = np.round(dbCKA_res, 2)
      plot_corrs(res2plot, domains_labels, 'CKA', plot_path, palette, width=7.48/3, height=10.63/4)


    # Load task and Create Full model
    domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']
    domains = {d: np.loadtxt('/home/laura.marras/Documents/Repositories/Action101/data/models/domains/group_ds_conv_{}.csv'.format(d), delimiter=',', skiprows=1)[:, 1:] for d in domains_list}
    full_model = {'full_model': np.hstack([domains[d] for d in domains_list])}
    full = full_model['full_model']
    #domains['full'] = full

    # Initialize arrays to save all permuted results 
    CCA_res = np.full((50, len(domains)), np.nan)
    dbCKA_res = np.full((50, len(domains)), np.nan)
  
    for d, domain in enumerate(list(domains.keys())):
      # Create permutation schema to shuffle columns of left-out domain for variance partitioning
      vperm_schema = permutation_schema(domains[domain].shape[0], n_perms=50, chunk_size=1)

      # For each permutation, build full model and shuffle columns of left-out domain
      for vperm in range(1, vperm_schema.shape[1]):
          vorder = vperm_schema[:, vperm] # first colum contains original order (non-permuted)                 
          shuffled = np.hstack([domains[dom] if dom!=domain else domains[dom][vorder,:] for dom in domains.keys()])
          
          # Run cca 
          CCA_res[vperm-1, d], _, _, _, _, _, _ = canonical_correlation(shuffled, full)
          
          # Run CKA
          dbCKA_res[vperm-1, d] = linear_cka(shuffled, full, debiasing=True)
          
    # Get average R2 results across perms
    CCA_res_avg = np.mean(CCA_res, axis=0)
    dbCKA_res_avg = np.mean(dbCKA_res, axis=0)

    labels = [dom + ' ' + str(domains[dom].shape[1]) for dom in list(domains.keys())]

    # Save results
    np.savez(path+'full_shuffled_correlation', cca=CCA_res_avg, cka=dbCKA_res_avg)

    if plot:
      barplot_corrs(np.vstack((CCA_res_avg, dbCKA_res_avg)), labels, ['CCA', 'CKA debiased'], path)

    print('d')