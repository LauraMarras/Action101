import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from utils.PalColormapImporter import read_pal_file, create_colormap_from_hex

def config_plt(textsize=8):

    plt.rc('font', family='Arial')
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=textsize)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)

    plt.rc('legend', fontsize=textsize)
    plt.rc('legend', loc='best')
    plt.rc('legend', frameon=False)

    plt.rc('grid', linewidth=0.5)
    plt.rc('axes', linewidth=0.5)
    plt.rc('xtick.major', width=0.5)
    plt.rc('ytick.major', width=0.5)

if __name__ == '__main__':
    config_plt()
    debiasing = True
    convolved = False

    # Load data
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/'
    domains_labels = ['space', 'effector', 'agent-object', 'social', 'emotion', 'linguistic', 'full']
    models_labels = np.array(['motion', 'relu6', 'power', 'relu5-1', 'gpt4'])
    domains_labels_short = ['space', 'effector', 'agent-\nobject', 'social', 'emotion', 'linguistic', 'full']
    
    dom_models = pd.read_csv('{}comp_models/CKA{}_doms{}_compmodels.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'), index_col=0)
    dom_doms = pd.read_csv('{}domains/CKA{}_btw_domains{}.csv'.format(path, '_db' if debiasing else '', '_conv' if convolved else '_bin'), index_col=0)
 
    dom_models.rename({key:domains_labels[k] for k, key in enumerate(dom_models.columns)}, axis=1, inplace=True)
    dom_doms.rename({key:domains_labels[k] for k, key in enumerate(dom_doms.columns)}, axis=1, inplace=True)
    dom_doms.rename({key:domains_labels[k] for k, key in enumerate(dom_doms.index)}, axis=0, inplace=True)

    data1 = np.round(dom_doms, 2)
    data2 = np.round(dom_models.loc[models_labels], 2).T

    # Create custom cmap from Afni palette file
    palette = '/data1/Action_teresi/CCA/code/reds_and_blues.pal'
    hex_colors = read_pal_file(palette)[1:128] # Select only red to blue part
    hex_colors.reverse()
    cmap = create_colormap_from_hex(hex_colors)

    # Dimensioni totali della figura
    fig_width = 4.986666666666667  # Larghezza totale della figura
    fig_height = 3  #2.67 Altezza totale della figura
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    gs = GridSpec(2, 2, height_ratios=[0.1, 9.9], width_ratios=[7,5], figure=fig, wspace=0.35, hspace=-0.2)

    cbarax = fig.add_subplot(gs[0,:])
    hm1 = fig.add_subplot(gs[1,0])
    hm2 = fig.add_subplot(gs[1,1])

    sns.heatmap(
        data1,
        vmin=0, vmax=1,
        annot=True,
        square=True,
        cbar=True,
        cmap=cmap,
        cbar_kws={'label': 'CKA', 'orientation':'horizontal'},
        ax=hm1,
        yticklabels=domains_labels_short,
        cbar_ax=cbarax,
        annot_kws={"size":8, "font":'arial'}
    )

    cbarax.xaxis.set_ticks_position('top')
    cbarax.xaxis.set_label_position('top')
    cbarax.xaxis.set_ticks([0,1])

    sns.heatmap(
        data2,
        vmin=0, vmax=1,
        annot=True,
        square=True,
        cbar=False,
        cmap=cmap,
        ax=hm2,
        yticklabels=domains_labels_short,
        annot_kws={"size":8, "font":'arial'}
    )
    hm1.set_xticklabels(hm1.get_xticklabels(), rotation=45, ha="right", rotation_mode='anchor')
    hm2.set_xticklabels(hm2.get_xticklabels(), rotation=45, ha="right", rotation_mode='anchor')

    cbarax.set_position([0.35, 0.87, 0.3, 0.02])
    cbarax.xaxis.labelpad = -10

    plot_path = '/home/laura.marras/Documents/Repositories/Action101/data/plots_final/'
    
    plt.savefig('{}CKA{}_domains{}_models.png'.format(plot_path, '_db' if debiasing else '', '_conv' if convolved else '_bin'))

    """
     # Plot
    dom_labels = ['space', 'effector', 'agent-object', 'social', 'emotion', 'linguistic', 'full']
    dbcka_df.rename({key:dom_labels[k] for k, key in enumerate(dbcka_df.columns)}, axis=1, inplace=True)
    models_new = np.array(models)[models_order]
    df = dbcka_df.loc[models_new]
    plot_path = '/home/laura.marras/Documents/Repositories/Action101/data/plots_final/'
    config_plt()

    palette = '/data1/Action_teresi/CCA/code/reds_and_blues.pal'
    
    # Round results to second decimal
    res2plot = np.round(df,2).T
    plot_heatmap(res2plot, dom_labels, df.index.tolist(), plot_path, 'CKA', palette, width=7.48/3, height=10.63/4)

    # barplot
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
    plt.savefig(models_path + 'cka{}.png'.format(binary))

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
    plt.savefig(models_path + 'cka_heatmap{}.png'.format(binary))
    """