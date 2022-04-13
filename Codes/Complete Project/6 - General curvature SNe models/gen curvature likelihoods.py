'''
Generalised curvature model, reading in chi^2 data, turning to likelihoods
getting confidence regions and plotting
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
import Codes.Module as module

# %%

# Likelihood and contour plots with our confidence finder

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\gen curvature chi^2 (300, 300).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
OL = np.linspace(0, 1.4, 500)
Om = np.linspace(0, 1, 500)

# set up plotting axis
Omgrid, OLgrid = np.meshgrid(Om, OL)

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# find confidences in likelihood with our own function for this:

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood, Omgrid, OLgrid, interp=1000)
likelihood *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood, Omgrid, OLgrid, accu=1000, interp=1000)

# plot as a contour map and heat map
fig4 = plt.figure()
ax4 = fig4.gca()
ax4.tick_params(labelsize=16)
ax4.set_ylabel('$\Omega_{m} $', fontsize=20)
ax4.set_xlabel('$\Omega_{\Lambda}$', fontsize=20)
contour_likelihood = ax4.contour(Omgrid, OLgrid, likelihood, heights, cmap=cm.jet)

heatmap = ax4.pcolormesh(Omgrid, OLgrid, likelihood)
fig4.colorbar(heatmap)
