
"""
Created on Mon Mar 14 09:49:00 2022

@author: bills
"""
'''
introduction of opacity as a parameter in chisquared while implicitly
marginalising over H0
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import Codes.Module as module

# %%
# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\chisquared including opacity(200 points).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 500)
epsil = np.linspace(-1,1,200)
epsil = epsil
z = np.linspace(np.min(df['z']), 1.8, 100)

# finding minimum of chisquared coords
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_epsil = epsil[index[1]]
print(min_Om, min_epsil)
# set up plotting axis
epsilgrid, Omgrid = np.meshgrid(epsil, Om)

# figure and visuals
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_ylabel('$\Omega_{m} $', fontsize=16)
ax1.set_xlabel('$\epsilon$', fontsize=16)
ax1.plot(min_epsil, min_Om, 'rx')  # minimum value pointer

# plot as heatmap and then add contours
heatmap = ax1.pcolormesh(epsilgrid, Omgrid, chisq_array)
contourplot = ax1.contour(epsilgrid, Omgrid, chisq_array, np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
#ax1.clabel(contourplot)
fig1.colorbar(heatmap)

# set up figure and plot a surface plot to observe the shape of chisquared
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_ylabel('$\Omega_{m} $', fontsize=16)
ax2.set_xlabel('$\epsilon$', fontsize=16)
ax2.set_zlabel('$\chi^2$', fontsize=16)
surf = ax2.plot_surface(epsilgrid, Omgrid, chisq_array, cmap=cm.jet)
fig2.colorbar(surf)

#%%

# contour plots with our new confidence finder

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\(60-80) redone for accurate chisq.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 100)

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_H0 = H0[index[1]]

# set up plotting axis
Hgrid, Omgrid = np.meshgrid(H0, Om)

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# find confidences in likelihood with our own function for this:

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood, Hgrid, Omgrid, interp=1000)
likelihood *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood, Hgrid, Omgrid, accu=1000, interp=1000)

# plot as a contour map and heat map
fig4 = plt.figure()
ax4 = fig4.gca()
ax4.tick_params(labelsize=16)
ax4.set_ylabel('$\Omega_{m} $', fontsize=20)
ax4.set_xlabel('$H_0 \ (m s^{-1} Mpc^{-1})$', fontsize=20)
contour_likelihood = ax4.contour(Hgrid/1000, Omgrid, likelihood, heights, cmap=cm.jet)
#ax4.clabel(contour_likelihood)

heatmap = ax4.pcolormesh(Hgrid/1000, Omgrid, likelihood)
fig4.colorbar(heatmap)


