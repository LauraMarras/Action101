import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


res_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/{}/group/single_doms/CCA_R2_group_singledoms.npz'
rois_path = '/home/laura.marras/Documents/Repositories/Action101/data/cca_results/{}/group/fullmodel/significantROIs_{}.txt'

conds = ['AV', 'vid', 'aud']
domains_list = ['space', 'movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']

# Plot 
fig, axs = plt.subplots(len(conds)+1, 1, figsize=(10,5*(len(conds)+1)), sharex=True)

# Plot cca and cka results
cca = np.load('/home/laura.marras/Documents/Repositories/Action101/data/models/full_shuffled_correlation.npz')['cca']
cka = np.load('/home/laura.marras/Documents/Repositories/Action101/data/models/full_shuffled_correlation.npz')['cka']

ax = axs[0]
bar_width = 0.2
x = np.arange(len(domains_list))

ax.bar(x - 0.5*bar_width, cca, bar_width, label='CCA')
ax.bar(x + 0.5 * bar_width, cka, bar_width, label='CKA')

# Labels and layout
ax.set_ylabel('Similarity')
ax.set_title('Average similarity between full model and shuffled model for each domain\nnumber of features for each domain')
ax.set_ylim(0.5,1)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Legend
ax.legend(title='Similarity', frameon=False)

# Add lineplot on number of columns for each domain
dom_len = [4, 12, 6, 8, 4, 4]

ax_right = ax.twinx()
ax_right.plot(x, dom_len, color='k', marker='o', linestyle='--')
ax_right.set_ylim(0,15)

ax_right.spines['top'].set_visible(False)
ax_right.set_ylabel('Features')


for c, cond in enumerate(conds):
    # Load results
    roisAV = np.loadtxt(rois_path.format('AV', 'AV')).astype(int)-1
    rois = np.loadtxt(rois_path.format(cond, cond)).astype(int)-1
    rois_indx = np.where(np.isin(roisAV, rois))[0]
    res = np.load(res_path.format(cond))['results_group_sd'][rois_indx,:]

    # Plot
    ax = axs[c+1]
    #sns.violinplot(res, ax=ax, inner='box')
    sns.boxplot(res, ax=ax)

    # Add title, labels etc
    ax.set_title(cond)
    ax.set_ylabel('R2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

ax.set_xlabel('Domains')
ax.set_xticklabels(domains_list, rotation=45)
plt.tight_layout()

fig.savefig('/home/laura.marras/Documents/Repositories/Action101/data/plots/violinplots.png')
fig.savefig('/home/laura.marras/Documents/Repositories/Action101/data/plots/boxplots.png')

