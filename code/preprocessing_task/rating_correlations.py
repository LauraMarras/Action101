import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jaccard
from scipy.linalg import lstsq, solve
from matplotlib import pyplot as plt
from itertools import combinations, permutations, product
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
import hoggorm as ho


# Define function to get correlation or Jaccard score
def get_corr(r1, r2, measure='jaccard'):
    # Initialize matrix
    corr_mat = np.zeros(r1.shape[1])

    # Iterate over features and measure correlation and Jaccard score
    for var_idx in range(r1.shape[1]):
        # Spearman
        if measure == 'spearman' or measure == 'Spearman':
            corr_mat[var_idx], _ = spearmanr(r1[:, var_idx], r2[:, var_idx])
        
        # Jaccard
        elif measure == 'jaccard' or measure == 'Jaccard':
            corr_mat[var_idx] = 1 - jaccard(r1[:, var_idx], r2[:, var_idx])

            # check if array of only zeros
            if np.max(r1[:, var_idx]) == 0 or np.max(r2[:, var_idx]) == 0:
                corr_mat[var_idx] = 0
    
    return corr_mat

# Def function to add numbers on top of bars
def gen_label(rects, as_int=False, no_zero=False, pvals=[], color=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for r, rect in enumerate(rects):
        height = rect.get_height()
        height_text = round(height, 2)
        if as_int:
            height_text = int(height)
        if height ==0:
            if no_zero:
                height_text =''
        if pvals:
            if pvals[r] <= 0.05:
                height_text = str(height_text)+'*'
            else:
                height_text = height_text
        if color:
            rect.set_color(color[r])
        
        ax = plt.gca()
        ax.annotate('{}'.format(height_text),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 3 points vertical offset
                    textcoords="offset points", size=8,
                    ha='center', va='bottom', rotation=60)

# Def function to plot correlations in single plots
def plot_single(dictionary, r1, r2, features, measure, downsample, color='#0081a7', save=True):
    
    # Create figure
    fig = plt.figure(figsize=(13,7))

    # Add bar
    correl = dictionary[(r1, r2)]
    plot = plt.bar(features, correl, color=color)
    plt.tight_layout()

    # Add xticks on the middle of the group bars
    plt.xlabel('Features', fontweight='bold', size=12)
    plt.ylabel(measure.capitalize(), fontweight='bold', size=12)
    plt.xticks(rotation=90, size=10)
    plt.yticks(np.arange(0, 1, 0.1), size=10)

    plt.ylim([0,1])

    ### Add labels on top of the bars
    gen_label(plot, as_int=False, no_zero=True, pvals=None, color=color)

    ### Remove spines
    for ax in fig.get_axes():     
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Set Title
    if r2 == 'group_2_3':
        r2_t = 'group model (2/3)'
    elif r2 == 'group_3_3':
        r2_t = 'group model (3/3)'
    else:
        r2_t = r2.upper() 
    plt.title('{} between {} and {}'.format(measure.capitalize(), r1. upper(), r2_t))

    # Save Figure
    if save:
        fig.savefig(out_path + '{}_btw_{}_{}_ds{}'.format(measure.capitalize(), r1.upper(), r2.upper(), downsample), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
        
# Def function to plot correlations in three subplots
def plot_triplette(dictionary, keys, features, measure, downsample, colors, save=True):
        
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(13,18))

    # Add bars
    for k, key in enumerate(keys):
        rects = axes[k].bar(features, dictionary[key], color=colors)
    
    # Add text on top of bars
        for r, rect in enumerate(rects):
            height = rect.get_height()
            height_text = round(height, 2)
            if height ==0:
                height_text =''
        
            axes[k].annotate('{}'.format(height_text),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 3 points vertical offset
                    textcoords="offset points", size=8,
                    ha='center', va='bottom', rotation=60)
            rect.set_color(colors[r])

    # Set Title
        if key[1] == 'group_2_3':
            t1 = 'group model (2/3)'
            ts = 'btw_raters_group2'
        elif key[1] == 'group_3_3':
            t1 = 'group model (3/3)'
            ts = 'btw_raters_group3'
        else:
            t1 = key[1].upper()
            ts = 'btw_raters'
        
        axes[k].set_title('{} between {} and {}'.format(measure.capitalize(), key[0].upper(), t1, downsample), )

    # Remove spines
    for ax in fig.get_axes():     
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(measure.capitalize(), fontweight='bold', size=12)
        ax.set_xticklabels(features, rotation=90)
        ax.set_yticks(np.arange(0,1.1,0.2))
        ax.set_ylim([0,1.1])

    # Set x label
    axes[2].set_xlabel('Features', fontweight='bold', size=12)

    plt.subplots_adjust(wspace=0, hspace=0.7)

    if save:
        # Save Figure
        fig.savefig(out_path + '{}_{}_ds{}'.format(measure.capitalize(), ts, downsample), dpi=300, facecolor='w')
    else:
        plt.show()

# Def function to plot sparsity
def plot_single_spars(r1, features, model = 'Group (2_3)', color='#0081a7', save=True):
    
    # Create figure
    fig = plt.figure(figsize=(13,7))

    # Add bar
    plot = plt.bar(features, r1, color=color)
    plt.tight_layout()

    # Add xticks on the middle of the group bars
    plt.xlabel('Features', fontweight='bold', size=12)
    plt.ylabel('Sparsity', fontweight='bold', size=12)
    plt.xticks(rotation=90, size=10)
    plt.yticks(np.arange(0, 1.1, 0.1), size=10)

    plt.ylim([0,1.1])

    ### Add labels on top of the bars
    gen_label(plot, as_int=False, no_zero=True, pvals=None, color=color)

    ### Remove spines
    for ax in fig.get_axes():     
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Set Title
    plt.title('Sparsity {}'.format(model))

    # Save Figure
    if save:
        fig.savefig(out_path + 'Sparsity/Sparsity_{}_dsbin'.format(model), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    plot = False
    plot_cc = True

    # Load files
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_PreprocessedTagging/tagging_carica101_'
    out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Results/Rating_Correlations/'

    all_down_sampled = np.load(data_path + 'all_down_sampled.npz')

    # Load features
    features = list(all_down_sampled['features'])

    # Define domains
    domains = { 'features':{
    'covariates': ['action_present', 'people_present', 'context_0', 'context_1', 'context_2', 'agent_H_NH', 'eff_visibility'],
    'emo': ['faces', 'ToM', 'EBL', 'EIA', 'simbolic_gestures', 'sociality'],
    'linguistic': ['dinamicity', 'durativity', 'telicity', 'iterativity', 'predictability'],
    'conceptual': ['multi_ag_vs_jointact_0', 'multi_ag_vs_jointact_1', 'multi_ag_vs_jointact_2', 'transitivity', 'tool_mediated', 'inter_scale'],
    'body': ['main_effector_0', 'main_effector_1', 'main_effector_2', 'main_effector_3', 'main_effector_4', 'main_effector_5', 'main_effector_6', 'main_effector_7', 'main_effector_8', 'main_effector_9', 'main_effector_10', 'main_effector_11', 'main_effector_12', 'main_effector_13'],
    'remaining': ['gesticolare', 'touch_0', 'touch_1', 'touch_2', 'target_0', 'target_1', 'target_2', 'target_3', 'target_4']
    }
    }

    # Re-order features
    order = []
    domains['feat_order'] = {}
    for key, val in domains['features'].items():
        domains['feat_order'][key[:3]+'_order'] = [features.index(c) for c in val]
        order += domains['feat_order'][key[:3]+'_order']

    features_reorder = [features[o] for o in order]

    # Domain Colors
    colors_domains = ['#5390d9' for c in domains['features']['covariates']] + ['#5e60ce' for c in domains['features']['emo']] + ['#6930c3' for c in domains['features']['linguistic']] + ['#e56b6f' for c in domains['features']['conceptual']] + ['#F19448' for c in domains['features']['body']] + ['#D4F292' for c in domains['features']['remaining']]
    
    # Create group models
    raters = ['lm', 'ai', 'lt']

    r1 = all_down_sampled['{}_ds_bin'.format(raters[0])][:, order]
    r2 = all_down_sampled['{}_ds_bin'.format(raters[1])][:, order]
    r3 = all_down_sampled['{}_ds_bin'.format(raters[2])][:, order]

    group_3_3 = r1*r2*r3
    group_2_3 = r1+r2+r3
    group_2_3[group_2_3<2] = 0
    group_2_3[group_2_3>1] = 1

    # Get correlations
    # Define downsampling and measure
    downsample = 'bin'
    measure = 'jaccard'

    # Initialize correlations dictionary
    correlations = {}

    # Get inter-subjects correlations
    raters_combs = list(combinations(raters, 2))
    for comb in raters_combs:
        r1 = all_down_sampled['{}_ds_{}'.format(comb[0], downsample)][:, order]    
        r2 = all_down_sampled['{}_ds_{}'.format(comb[1], downsample)][:, order]
        
        correlations[comb] = get_corr(r1, r2, measure)

    # Get correlation between subject and group models
    for rater in raters:
        r1 = all_down_sampled['{}_ds_{}'.format(rater, downsample)][:, order]
        correlations[(rater, 'group_3_3')] = get_corr(r1, group_3_3, measure)
        correlations[(rater, 'group_2_3')] = get_corr(r1, group_2_3, measure)

    # Calculate sparsity
    r1_sparsity = 1 - np.count_nonzero(r1, 0)/ float(r1.shape[0])
    r2_sparsity = 1 - np.count_nonzero(r2, 0)/ float(r2.shape[0])
    r3_sparsity = 1 - np.count_nonzero(r3, 0)/ float(r3.shape[0])
    group_2_sparsity = 1 - np.count_nonzero(group_2_3, 0)/ float(group_2_3.shape[0])
    group_3_sparsity = 1 - np.count_nonzero(group_3_3, 0)/ float(group_3_3.shape[0])
        
    # Create domain indices and split matrix into many domain matrices
    domains['feat_idx'] = {}
    domains['matrices'] = {'group_2_3':{}, 'group_3_3':{}}

    for key, val in domains['features'].items():   
        indices = [features_reorder.index(c) for c in val]
        domains['feat_idx'][key[:3]+'_idx'] = indices
        domains['matrices']['group_2_3'][key[:3]+'_mat'] = group_2_3[:, indices]
        domains['matrices']['group_3_3'][key[:3]+'_mat'] = group_3_3[:, indices]

    # Calculate correlation between domains
    domains['correlations'] = {'covariates':{}, 'emo':{}, 'linguistic':{}, 'conceptual':{}, 'body':{}, 'remaining':{}}
    n_dom = len(domains['features'].keys())
    domains_combs = list(product(domains['matrices']['group_2_3'].keys(), repeat=2))
    r2_mat = np.zeros((n_dom,n_dom))
    rv_mat = np.zeros((n_dom,n_dom))
    rv_mat_man = np.zeros((n_dom,n_dom))

    for c, comb in enumerate(domains_combs):
        x = domains['matrices']['group_2_3'][comb[0]]
        y = domains['matrices']['group_2_3'][comb[1]]

        key_y = [x for x in domains['features'].keys() if comb[0][:3] in x][0]
        ylabs = domains['features'][key_y]
        key_x = [x for x in domains['features'].keys() if comb[1][:3] in x][0]
        xlabs = domains['features'][key_x]
        
        # Canonical correlation
        n_components = np.min([x.shape[1], y.shape[1]])
        cca = CCA(n_components = n_components)
        cca.fit(x,y)
        rscore = cca.score(x,y)
        U, V = cca.transform(x,y)
        A = cca.x_weights_
        B = cca.y_weights_

        # Vettorizziamo
        U_v = U.flatten('F')
        V_v = V.flatten('F')

        # prediciamo V partendo da U
        b_coeffs = lstsq(U, V)[0]

        V_predic = np.dot(U, b_coeffs)
        
        Y_predic = np.dot(V_predic, np.linalg.inv(B))

        SSres = np.sum((y-Y_predic)**2)
        SStot=np.sum((y - np.mean(y))**2)
        r2=1-(SSres/SStot)
        
        row = int(c%n_dom)
        col = int((c-(c%n_dom))/n_dom)

        r2_mat[row, col] = r2
        domains['correlations'][key_y][key_x] = r2

        # RV coefficient
        # Center data
        x_c = x - np.mean(x, axis=0)
        y_c = y - np.mean(y, axis=0)
        
        # Calculate adjusted RV coefficient using hoggorm library
        rv_c = ho.RV2coeff([x_c, y_c])[0,1]
        rv_mat[row, col] = rv_c

        # Do it manually
        XX = np.dot(x_c, x_c.T)
        YY = np.dot(y_c, y_c.T)

        XX0 = XX - np.diag(np.diag(XX))
        YY0 = YY - np.diag(np.diag(YY))

        rssx = np.sqrt(np.sum(XX0**2))
        rssy = np.sqrt(np.sum(YY0**2))

        adjusted_rv = np.trace(np.dot(XX0, YY0))/rssx/rssy
        rv_mat_man[row, col] = adjusted_rv


    if plot_cc:
        # Plot correlation matrices
        fig = plt.figure(figsize = (n_dom, n_dom))
        hm = sns.heatmap(r2_mat, cmap='coolwarm', annot=True, vmin= -1, vmax=1, linewidths=1,  cbar_kws={'label': 'R2 cca'})
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.set_yticklabels(list(domains['features'].keys()), rotation=0)
        ax.set_xticklabels(list(domains['features'].keys()), rotation=90)
        fig.savefig(out_path + 'CanonicalCorr/R2_man_group2', dpi=300, facecolor='w', bbox_inches='tight')

    
    if plot:
        # Plot correlations 
        colors = ['#5390d9', '#5e60ce', '#6930c3', '#eaac8b', '#e56b6f', '#b56576', '#a1c181', '#619b8a', '#31572c']
        
        # in single plots
        for c, comb in enumerate(correlations.keys()):
            r1 = comb[0]
            r2 = comb[1]
            #plot_single(correlations, r1, r2, features, measure, downsample, color=colors_domains, save=True)
        
        # in three subplots
        #plot_triplette(correlations, [('lm', 'ai'), ('lm', 'lt'), ('ai', 'lt')], features, measure, downsample, colors=colors_domains, save=False)

        # plot sparsity
        plot_single_spars(group_2_sparsity, features_reorder, 'Group (2_3)', color=colors_domains, save=True)
        plot_single_spars(r1_sparsity, features_reorder, 'LM', color=colors_domains, save=True)
        plot_single_spars(r2_sparsity, features_reorder, 'AI', color=colors_domains, save=True)
        plot_single_spars(r3_sparsity, features_reorder, 'LT', color=colors_domains, save=True)