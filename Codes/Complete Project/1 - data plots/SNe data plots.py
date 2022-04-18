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
df = pd.read_excel('Codes\\Complete Project\\Datasets\\SNe data.xlsx')
df = df.sort_values('z')  # sort in increasing z

# set up figure and visuals for intial data plot
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$Redshift \ z$', fontsize=20)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=20)

# plot the data with errorbars
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=3, ecolor='k')


# convert data from mu to d_L, make it a new column
df_dL = 10**(0.2*df['mu'] - 5)
df.insert(4, 'dL Mpc', df_dL)

# propagate errors and add as new column to data frame
ddL = 0.2*np.log(10)*10**(0.2*df['mu'] - 5) * df['dmu']
df.insert(5, 'ddL Mpc', ddL)

# plot dL data with errorbars:
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_xlabel(r'$Redshift \ z$', fontsize=20)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=20)

# plot the data as errorbar plot
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=4, ecolor='k')
