import numpy as np
from matplotlib import pyplot as plt

# load results

r01 = np.load('results_n01.npy')
r05 = np.load('results_n05.npy')
r1 = np.load('results_n1.npy')
r2 = np.load('results_n2.npy')

plt.hist(r01)
