import numpy as np
import pandas as pd
from scipy.spatial.distance import dice
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from itertools import combinations

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

def plot_actionpres(mat, bar_values, raters, path):

    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[7,1], wspace=50)

    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax.imshow(mat, aspect='auto', cmap='Purples', interpolation='nearest')

    # Set axis labels
    ax.set_yticks([*range(len(raters))])
    ax.set_yticklabels(raters)
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

if __name__ == '__main__':

    path = '/home/laura.marras/Documents/Repositories/Action101/data/models/inter_raters_agreement/'
    raters = ['AI', 'LM', 'LT']
    pairings = list(combinations(raters, 2))

    # Load single tagger action_present column
    ap_dict = {r: pd.read_csv(path + '{}_bin_ds.csv'.format(r), sep=',')['action_present'].to_numpy() for r in raters}
    
    # Initialize results matrix
    dice_res = np.full((len(pairings)), np.nan)

    for p, pair in enumerate(pairings):
        dice_res[p] = 1-dice(ap_dict[pair[0]], ap_dict[pair[1]])
        
    # Plot 
    ap_mat = np.vstack(list(ap_dict.values()))
    plot_actionpres(ap_mat, dice_res, raters, path)
