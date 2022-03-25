'''
First plots of H(z) data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

# Read in data as pandas dataframe
df = pd.read_excel('data\\Hz\\Wei Hz data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=18)
ax.set_xlabel(r'$z$', fontsize=20)
ax.set_ylabel(r'$H(z) \ kms^{-1}Mpc^{-1}$', fontsize=20)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['Hz'], yerr=df['dHz'],
            capsize=2, fmt='.', markersize=5, ecolor='k')
