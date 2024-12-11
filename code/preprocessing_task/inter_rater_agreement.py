import numpy as np
import pandas as pd
from utils.similarity_measures import canonical_correlation, linear_cka, gram_linear, center_gram, cka
from utils.permutation_schema_func import permutation_schema
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import dice
import matplotlib.gridspec as gridspec

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

def plot_actionpres(ai,lm,lt, bar_values, path):

    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[7,1], wspace=50)

    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax.imshow(np.vstack((ai,lm,lt)), aspect='auto', cmap='Purples', interpolation='nearest')

    # Set axis labels
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['AI', 'LM', 'LT'])
    ax.set_ylabel('Raters')

    ax.axhline(0.5, color='w', linewidth=2)
    ax.axhline(1.5, color='w', linewidth=2)

    run_cuts = np.array([0, 268,  493,  813, 1138, 1374, 1614])
    for r in run_cuts:
        ax.axvline(r, color='r', linewidth=2)

    ax.set_xlim([-1.5,1615.5])
    ax.set_xlabel('Time (s)')

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(run_cuts[:-1])
    ax2.set_xticklabels(['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Run6'])

    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add legend
    legend_elements = [
    mpatches.Patch(facecolor=plt.cm.Purples(0.99), edgecolor='black', label='present'),
    mpatches.Patch(facecolor=plt.cm.Purples(0), edgecolor='black', label='not present')
    ]
    ax.legend(handles=legend_elements, loc='upper left', title="Action", frameon=False, bbox_to_anchor=(1,1))

    # Add barplot
    xbar = [0,0.5,1]
    ax_bar.bar(xbar, bar_values, width=0.5, color=['#058c42', '#ec4d37', '#6f2dbd'], alpha=0.85)
    ax_bar.set_ylabel('Dice similarity')
    ax_bar.set_xlabel('Raters pair')

    ax_bar.set_xticks(xbar)
    ax_bar.set_xticklabels(['','',''])

    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)

    labelss = ['AI vs LM', 'AI vs LT', 'LM vs LT']
    legend_elements2 = [
    mpatches.Patch(facecolor='#058c42', label='AI vs LM'), mpatches.Patch(facecolor='#ec4d37', label='AI vs LT'), mpatches.Patch(facecolor='#6f2dbd', label='LM vs LT'),
    ]

    ax_bar.legend(handles=legend_elements2, loc='upper center', title="Raters pair", frameon=False, bbox_to_anchor=(-1.3, 0.5))

    gs.tight_layout(fig)
    plt.savefig(path + 'inter-rater_action_present.png')

def plot_corrs(results, results2, domains, measure, path):
   
    # Parameters
    n_rater_pairs = results.shape[0]
    n_domains = results.shape[1]
    rater_pairs = ['AI vs LM', 'AI vs LT', 'LM vs LT']
    bar_width = 0.2

    x = np.arange(n_domains)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = ['#058c42', '#ec4d37', '#6f2dbd']

    for i in range(n_rater_pairs):
        ax.bar(x + i * bar_width, results[i], bar_width, label=rater_pairs[i],  color=colors[i], alpha=0.85)
        
        ax.bar(x[:-1] + i * bar_width, results2[i,:-1], bar_width, color=colors[i], hatch='////', edgecolor='black')

    ax_right = ax.twinx()
    ax_right.set_ylim(0,1)
    ax_right.spines['top'].set_visible(False)
    ax_right.set_ylabel('Dice similarity')


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
    solid_patches = [mpatches.Patch(color=colors[i], label=f"{rater_pairs[i]}") for i in range(n_rater_pairs)]
    hatch_patches = [mpatches.Patch(facecolor='w', hatch='////', edgecolor='black', label='Shuffled'), mpatches.Patch(facecolor='k', edgecolor='black', label='Real')]
    ax.legend(handles=solid_patches + hatch_patches, title='Rater Pairs', frameon=False)

    # Save
    plt.savefig(path + 'inter-rater_{}.png'.format(measure))

def plot_corrs_means(results, results2, res_std, res2_std, domains, measure, path):
   
    # Parameters
    n_domains = results.shape[0]
    bar_width = 0.6

    x = np.arange(n_domains)

    # Plot
    inchsize_size = tuple(np.array([21/3, 29.7/4])*0.393701)

    # Dimensioni totali della figura
    fig_width = 4.986666666666667 / 2 # Larghezza totale della figura
    fig_height = 3  #2.67 Altezza totale della figura
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300) #(2.5, 3.5)
    #colors = ['#058c42', '#ec4d37', '#6f2dbd']
    colors = ['#548ba8','#719e49','#e75e5d', '#e79c3c', '#866596', '#77452d', '#0d1b2a']

    ax.bar(x, results, bar_width, color=colors, alpha=0.85, yerr=res_std)
    ax.bar(x, results2, bar_width, color=colors, hatch='////', edgecolor='black', yerr=res2_std, error_kw=dict(elinewidth=0.5))

    # Labels and layout
    #ax.set_xlabel('Domains')
    ax.set_ylabel('{}'.format(measure))
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylim(0,1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Legend
    solid_patches = [mpatches.Patch(color=colors[c], label=domains[c]) for c in range(len(domains))]
    ax.legend(handles=solid_patches, loc='upper right', ncol=1, handlelength=1, labelspacing=0, handletextpad=0.3, columnspacing=0.4)

    # Save
    plt.savefig(path + '{}.png'.format(measure.replace(' ', '_')))

if __name__ == '__main__':

    # Define parameters
    taggers = ['LM', 'AI', 'LT']
    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/inter_raters_agreement/'
    
    domains = {
    'space':                        ['context_0', 'context_1', 'context_2', 'inter_scale'],

    'movement':                     ['eff_visibility', 'main_effector_0', 'main_effector_1', 'main_effector_2', 
                                    'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6',
                                    'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10'],

    'agent_objective':              ['target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2'],

    'social_connectivity':          ['sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 
                                     'multi_ag_vs_jointact_2', 'ToM', 'people_present'],

    'emotion_expression':           ['EBL', 'EIA', 'gesticolare', 'simbolic_gestures'],

    'linguistic_predictiveness':    ['durativity', 'telicity', 'iterativity', 'dinamicity'],

    'full':                         ['context_0', 'context_1', 'context_2', 'inter_scale', 
                                    
                                    'eff_visibility', 'main_effector_0', 'main_effector_1', 'main_effector_2', 
                                    'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6',
                                    'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10', 
                                    
                                    'target_0','target_1', 'agent_H_NH', 'tool_mediated', 'transitivity', 'touch_2', 
                                    
                                    'sociality', 'touch_1', 'target_2', 'target_3', 'multi_ag_vs_jointact_1', 
                                    'multi_ag_vs_jointact_2', 'ToM', 'people_present',
                                    
                                    'EBL', 'EIA', 'gesticolare', 'simbolic_gestures',
                                    
                                    'durativity', 'telicity', 'iterativity', 'dinamicity']
    } # 'action_present':               'action_present'
    
    # Load single tagger tagging
    AI = pd.read_csv(path + '{}_ds.csv'.format('AI'), sep=',')
    LM = pd.read_csv(path + '{}_ds.csv'.format('LM'), sep=',')
    LT = pd.read_csv(path + '{}_ds.csv'.format('LT'), sep=',')
    
    # Create perm schema
    max_cols = np.max([len(v) for v in domains.values()])
    t_points = AI.shape[0]
    n_perms = 1000
    perm_schema = permutation_schema(t_points, n_perms)
    
    # Initialize results matrix
    res = np.zeros((len(taggers), len(domains), n_perms+1))
    
    # Calculate correlation between raters for each domain and permutation
    for d, dom in enumerate(domains.keys()):
        ai = AI[domains[dom]].to_numpy()
        lm = LM[domains[dom]].to_numpy()
        lt = LT[domains[dom]].to_numpy()
        
        for p in range(perm_schema.shape[1]):
            order = perm_schema[:,p]

            # For all domains get CKA
            if dom != 'action_present':
                res[0, d, p] = linear_cka(ai, lm[order, :], debiasing=True)
                res[1, d, p] = linear_cka(ai, lt[order, :], debiasing=True)
                res[2, d, p] = linear_cka(lm, lt[order, :], debiasing=True)
            
            # For Action present get dice similarity (do not permute - sparcity too low)
            else:
                if p==0:
                    res[0, d, p] = 1-dice(ai, lm[order])
                    res[1, d, p] = 1-dice(ai, lt[order])
                    res[2, d, p] = 1-dice(lm, lt[order])
                
                else:
                    continue

    # Get null distribution 99th percentile
    res_null = np.percentile(res[:, :, 1:], q=99, axis=-1)
    res_real = res[:, :, 0]

    # Get group mean and stds
    res_mean = np.mean(res_real, axis=0)
    res_null_mean = np.mean(res_null, axis=0)
    res_std = np.std(res_real, axis=0)
    res_null_std = np.std(res_null, axis=0)

    # Plot
    config_plt()
    domains_labels = ['space', 'effector', 'agent-object', 'social', 'emotion', 'linguistic', 'full']
    plot_path = '/home/laura.marras/Documents/Repositories/Action101/data/plots_final/'
    plot_corrs_means(res_mean, res_null_mean, res_std, res_null_std, domains_labels, 'Inter-rater agreement (CKA)', plot_path)


    act_ai = AI['action_present'].to_numpy()
    act_lm = LM['action_present'].to_numpy()
    act_lt = LT['action_present'].to_numpy()
    plot_actionpres(act_ai, act_lm, act_lt, path)

    print('d')