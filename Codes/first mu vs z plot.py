'''
Here we plot the first representation of the data
a plot of mu vs z (distance modulus against redshift)
'''

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df.sort_values('z')

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=16)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

