import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

data_path = '/home/laura.marras/Documents/Repositories/Action101/data/preprocessed_tagging/tagging_carica101_'
df_all = pd.read_csv(data_path + 'all_preprocessed.csv', sep=',')
df_all['duration_seconds'] = (df_all.offset_ds - df_all.onset_ds)/10

ai_dur = df_all[df_all.rater == 'AI'].duration_seconds.to_numpy()
lm_dur = df_all[df_all.rater == 'LM'].duration_seconds.to_numpy()
lt_dur = df_all[df_all.rater == 'LT'].duration_seconds.to_numpy()
all_dur = df_all.duration_seconds.to_numpy()

fig = plt.figure(figsize=(6, 4), dpi=300)
colors = ['#058c42', '#ec4d37', '#6f2dbd']
    
plt.hist(x=[ai_dur, lm_dur, lt_dur], bins=100, range=(0,90), histtype='barstacked', color=colors, label=['AI', 'LM', 'LT'])
plt.xlabel('Duration (seconds)',)
plt.xticks(np.arange(0, 91, 5))
plt.ylabel('Event count')

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False, loc='upper center')

plt.savefig('hist.png')

print('')