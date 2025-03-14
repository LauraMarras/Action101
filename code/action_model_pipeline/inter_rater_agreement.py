import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

def linear_cka(X, Y, debiasing=False):
    
    """
    Compute CKA with a linear kernel, in feature space.
    Inputs:
    - X: array, 2d matrix of shape = samples by features
    - Y: array, 2d matrix of shape = samples by features
    - debiasing: bool, whether to apply debiasing or not; default = False 
    
    Output:
    - CKA: float, the value of CKA between X and Y
    """

    # Recenter
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    # Get dot product similarity and normalized matrices
    similarity = np.linalg.norm(np.dot(X.T, Y))**2 # Squared Frobenius norm of dot product between X transpose and Y
    normal_x = np.linalg.norm(np.dot(X.T, X)) 
    normal_y = np.linalg.norm(np.dot(Y.T, Y))
    
    # Apply debiasing
    if debiasing: 
      n = X.shape[0]
      bias_correction_factor = (n-1)*(n-2)
    
      SS_x = np.sum(X**2, axis=1) # Sum of squared rows 
      SS_y = np.sum(Y**2, axis=1)
      Snorm_x = np.sum(SS_x) # Squared Frobenius norm
      Snorm_y = np.sum(SS_y)
      
      similarity = similarity - ((n/(n-2)) * np.dot(SS_x, SS_y)) + ((Snorm_x * Snorm_y) / bias_correction_factor)
      normal_x = np.sqrt(normal_x**2 - ((n/(n-2)) * np.dot(SS_x, SS_x)) + ((Snorm_x * Snorm_x) / bias_correction_factor)) 
      normal_y = np.sqrt(normal_y**2 - ((n/(n-2)) * np.dot(SS_y, SS_y)) + ((Snorm_y * Snorm_y) / bias_correction_factor))

    # Get CKA between X and Y
    CKA = similarity / np.dot(normal_x, normal_y)

    return CKA

def permutation_schema(n_tpoints, n_perms=1000, chunk_size=15, seed=0, flip=True):
    
    """
    Creates an index permutation schema
    
    Inputs:
    - n_tpoints : int, number of timepoints (rows)
    - n_perms : int, number of permutations (columns); default = 1000
    - chunk_size: int, size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: int, seed for the random permutation; default = 0
    - flip: bool, decide wether to flip or not random number of chunks, default = True

    Outputs:
    - perm_schema : array, 2d matrix of shape = n_tpoints by n_perms + 1; first row contains unshuffled indices
    """

    # Set seed
    np.random.seed(seed)
    
    # Create an array of indices from 0 to number of timepoints
    indices = np.arange(n_tpoints)

    # Initialize the permutation schema matrix
    perm_schema = np.zeros((n_tpoints, n_perms+1), dtype=int)

    # Create chunks of contiguous timepoints
    chunks = [indices[i:i + chunk_size] for i in range(0, n_tpoints, chunk_size)]

    # Flip some of the chunks
    if flip:
        flip_inds = np.random.choice([0, 1], size=len(chunks))
        chunks = [np.flip(chunk) if flip_inds[c] == 1 else chunk for c, chunk in enumerate(chunks)]

    # Shuffle the order of the chunks separately for each permutation
    for i in range(1,n_perms+1):
        perm_schema[:,i] = np.concatenate([list(chunks[c]) for c in np.random.permutation(len(chunks))])
        
    # Assign original indices to first column of permutation schema
    perm_schema[:,0] = indices

    return perm_schema

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

def plot_corrs_means(results, results2, res_std, res2_std, domains, fname, path):
   
    # Parameters
    n_domains = results.shape[0]
    bar_width = 0.6

    x = np.arange(n_domains)

    # Plot
    fig_width = 4.986666666666667 / 2 
    fig_height = 3 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    colors = ['#548ba8','#719e49','#e75e5d', '#e79c3c', '#866596', '#77452d', '#0d1b2a']

    ax.bar(x, results, bar_width, color=colors, alpha=0.85, yerr=res_std)
    ax.bar(x, results2, bar_width, color=colors, hatch='////', edgecolor='black', yerr=res2_std, error_kw=dict(elinewidth=0.5))

    # Labels and layout
    ax.set_ylabel('Inter-rater agreement (CKA)')
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
    plt.savefig(path + '{}.png'.format(fname))

if __name__ == '__main__':

    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/inter_raters_agreement/'
    plot = True
    debiasing = True
    convolved = False
    
    # Load CKA results
    try:
        res = np.load('{}inter_rater_CKA{}_{}_permut.npy'.format(path, '_db' if debiasing else '', 'conv' if convolved else 'bin'))
    
    except FileNotFoundError:
        
        raters = ['LM', 'AI', 'LT']
        pairings = list(combinations(raters, 2))
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
        tagging_dict = {r: pd.read_csv(path + '{}_{}_ds.csv'.format(r, 'conv' if convolved else 'bin'), sep=',') for r in raters}
        
        # Create perm schema
        max_cols = np.max([len(v) for v in domains.values()])
        t_points = tagging_dict[raters[0]].shape[0]
        n_perms = 1000
        perm_schema = permutation_schema(t_points, n_perms)
        
        # Initialize results matrix
        res = np.zeros((len(raters), len(domains), n_perms+1))
        
        # Calculate correlation between raters for each domain and permutation
        for d, dom in enumerate(domains.keys()):
            
            for p in range(perm_schema.shape[1]):
                order = perm_schema[:,p]

                # For all domains get CKA
                for pr, pair in enumerate(pairings):
                    t1 = tagging_dict[pair[0]][domains[dom]].to_numpy()
                    t2 = tagging_dict[pair[1]][domains[dom]].to_numpy()
                    
                    res[pr, d, p] = linear_cka(t1, t2[order, :], debiasing=debiasing)

        np.save('{}inter_rater_CKA{}_{}_permut.npy'.format(path, '_db' if debiasing else '', 'conv' if convolved else 'bin'), res)
    
    # Get null distribution 99th percentile
    res_null = np.percentile(res[:, :, 1:], q=99, axis=-1)
    res_real = res[:, :, 0]

    # Get group mean and stds
    res_mean = np.mean(res_real, axis=0)
    res_null_mean = np.mean(res_null, axis=0)
    res_std = np.std(res_real, axis=0)
    res_null_std = np.std(res_null, axis=0)

    # Plot
    if plot:
        config_plt()
        domains_labels = ['space', 'effector', 'agent-object', 'social', 'emotion', 'linguistic', 'full']
        fname = 'inter_rater_CKA{}_{}'.format('_db' if debiasing else '', 'conv' if convolved else 'bin')
        plot_corrs_means(res_mean, res_null_mean, res_std, res_null_std, domains_labels, fname, path)