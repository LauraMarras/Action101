import numpy as np
from matplotlib import pyplot as plt

# load results
datapath = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Action101/Results/Correlations_debugging/'
noises = {'{}'.format(noise): np.reshape(np.load(datapath + 'results_n{}.npy'.format(noise))[1,1:,:], (500, 5)) for noise in ['01', '05', '1', '2']}

labels = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']

# Show null distribution of 
for key, val in noises.items():
    max = np.max(np.histogram(val, 500)[1])
    
    min = np.min(np.histogram(val, 500)[1])
    xticks = [round(x,4) for x in np.linspace(-0.06606812653002891, -0.04768910959680839, 10)]
    fig = plt.figure()
    plt.hist(val, 500, histtype='step', label=labels)
    if len(key) > 1:
        tit = key[0] + '.' + key[1]
    else:
        tit = key
    plt.title('Noise coefficient = {}'.format(tit))
    
    plt.xticks(xticks, rotation=90, fontsize=6)
    plt.legend(frameon=False)

    plt.xlabel('R2')
    plt.ylabel('Count')

    plt.xlim(-0.06606812653002891, -0.04768910959680839)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(datapath + '{}nulldist_roi2'.format(key), dpi=300)