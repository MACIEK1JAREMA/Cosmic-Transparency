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
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\chisq(Om, eps) (500 points).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 0.6, 500)
epsil = np.linspace(-0.3, 0.3, 500)
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
ax1.tick_params(labelsize=16)
ax1.set_ylabel('$\Omega_{m} $', fontsize=20)
ax1.set_xlabel('$\epsilon$', fontsize=20)
ax1.plot(min_epsil, min_Om, 'rx')  # minimum value pointer

# plot as heatmap and then add contours
heatmap = ax1.pcolormesh(epsilgrid, Omgrid, chisq_array)
contourplot = ax1.contour(epsilgrid, Omgrid, chisq_array, np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
cbar = fig1.colorbar(heatmap)
cbar.ax.tick_params(labelsize=16)
