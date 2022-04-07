'''
Our first attempt at combining the two datasets

Read in both chi^2 files, get likelihoods,
find confidence regions in 2D SNe results, plot them,
then find them on H(z) 1D χ^2 extend them over the whole
range of ϵ and plot them on top.
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import Codes.Module as module


# %%

# ############################################################################
# SNe data
# ############################################################################

# Read in generated array
chisq_df = pd.read_excel('data\\Final datasets\\chisq with opacity corrected (500 points).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# set up the model axis
eps = np.linspace(-0.3, 0.3, 500)
Om = np.linspace(0, 0.6, 500)

# finding minimum of chisquared coords
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_eps = eps[index[1]]

# set up plotting axis
epsgrid, Omgrid = np.meshgrid(eps, Om)

# switch to likelihoods
likelihood_SNe = np.exp((-chisq_array**2)/2)

# find confidences in likelihood with our own function for this:

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood_SNe, epsgrid, Omgrid)
likelihood_SNe *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood_SNe, epsgrid, Omgrid, accu=1000)

# plot as a contour map only
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_ylabel('$\Omega_{m} $', fontsize=20)
ax.set_xlabel('$\epsilon$', fontsize=20)
ax.contour(epsgrid, Omgrid, likelihood_SNe, heights, cmap=cm.jet)

# ############################################################################
# H(z) data
# ############################################################################

# Read in generated array
chisq_df = pd.read_excel('data\\Hz\\2D chisquared for H(z) 500.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# prepare grids and chi^2
Om = np.linspace(0, 0.6, 500)
epsgrid, Omgrid = np.meshgrid(eps, Om)
chisq_array -= np.min(chisq_array)

# likelihood
like = np.exp(-(chisq_array**2)/2)

# flat prior marginalising

# marginalising over Om - with flat prior:
lik_margin = np.sum(like, axis=1)

# extend over epsilon
likelihood_Hz = np.tile(lik_margin, (len(eps), 1)).transpose()

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood_Hz, epsgrid, Omgrid)
likelihood_Hz *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood_Hz, epsgrid, Omgrid, accu=1000)

# plot as a contour map only
ax.contour(epsgrid, Omgrid, likelihood_Hz, heights, cmap=cm.jet)

# %%

'''
Multiplying likelihoods together for overall constraint
'''

# ############################################################################
# SNe data
# ############################################################################

# Read in generated array
chisq_df = pd.read_excel('data\\Final datasets\\chisq with opacity corrected (500 points).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# set up the model axis
eps = np.linspace(-0.3, 0.3, 500)
Om = np.linspace(0, 0.6, 500)

# finding minimum of chisquared coords
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_eps = eps[index[1]]

# set up plotting axis
epsgrid, Omgrid = np.meshgrid(eps, Om)

# switch to likelihoods
likelihood_SNe = np.exp((-chisq_array**2)/2)

# ############################################################################
# H(z) data
# ############################################################################

# Read in generated array
chisq_df = pd.read_excel('data\\Hz\\2D chisquared for H(z) 500.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# prepare grids and chi^2
Om = np.linspace(0, 0.6, 500)
epsgrid, Omgrid = np.meshgrid(eps, Om)
chisq_array -= np.min(chisq_array)

# likelihood
like = np.exp(-(chisq_array**2)/2)

# flat prior marginalising

# marginalising over Om - with flat prior:
lik_margin = np.sum(like, axis=1)

# extend over epsilon
likelihood_Hz = np.tile(lik_margin, (len(eps), 1)).transpose()


# ############################################################################
# Combine
# ############################################################################

likelihood_all = likelihood_Hz * likelihood_SNe

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood_all, epsgrid, Omgrid)
likelihood_all *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood_all, epsgrid, Omgrid, accu=1000)

# plot as a contour plot and heat map
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_ylabel('$\Omega_{m} $', fontsize=20)
ax1.set_xlabel('$\epsilon$', fontsize=20)
ax1.contour(epsgrid, Omgrid, likelihood_all, heights, cmap=cm.jet)

heatmap = ax1.pcolormesh(epsgrid, Omgrid, likelihood_all)
cbar = fig1.colorbar(heatmap)
cbar.ax.tick_params(labelsize=16)

# %%

'''
Marginalising over Om with flat prior to constrain opacity
NOTE --- only run once cell above has been run
'''

# marginalising over Om - with flat prior:
lik_margin = np.sum(likelihood_all, axis=0)

# normalise to sum=1 for scip rvdiscrete.
lik_margin /= np.sum(lik_margin)

# set up figure + visuals and plot it:
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=18)
ax2.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ and \ \Omega_{m} \ L(\epsilon)$', fontsize=20)
ax2.set_xlabel(r'$\epsilon$', fontsize=20)
ax2.plot(eps, lik_margin)

# find peak value and where 68.3% of it lies for 1 \sigma error
eps_found = eps[np.where(lik_margin == np.max(lik_margin))[0]]

variables = st.rv_discrete(values=(eps, lik_margin))
confidence1 = variables.interval(0.683)[1] - eps_found
confidence2 = eps_found - variables.interval(0.683)[0]

print(f'eps = {round(eps_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')


