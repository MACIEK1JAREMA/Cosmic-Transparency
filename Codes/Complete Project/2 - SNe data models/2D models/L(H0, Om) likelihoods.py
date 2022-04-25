
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import Codes.Module as module

# %%

# plot found $\chi^2$ from excel first

# Read in Sne data as pandas dataframe
df = pd.read_excel('Codes\\Complete Project\\Datasets\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\(60-80) chisq(Om, H0).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 500)

# finding minimum of chisquared coords
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_H0 = H0[index[1]]
print(min_Om, min_H0)
# set up plotting axis
Hgrid, Omgrid = np.meshgrid(H0, Om)

# figure and visuals
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_ylabel('$\Omega_{m} $', fontsize=20)
ax1.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$', fontsize=20)
ax1.plot(min_H0/1000, min_Om, 'rx')  # minimum value pointer

# plot as heatmap and then add contours
heatmap = ax1.pcolormesh(Hgrid/1000, Omgrid, chisq_array)
contourplot = ax1.contour(Hgrid/1000, Omgrid, chisq_array, np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
# used 2-parameter \Delta \chi^2 values.
fig1.colorbar(heatmap)

# %%

# Likelihood plot with our confidence finder

# Read in Sne data as pandas dataframe
df = pd.read_excel('Codes\\Complete Project\\Datasets\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\(60-80) chisq(Om, H0).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 500)

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
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=16)
ax2.set_ylabel('$\Omega_{m} $', fontsize=20)
ax2.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$', fontsize=20)
contour_likelihood = ax2.contour(Hgrid/1000, Omgrid, likelihood, heights, cmap=cm.jet)
#ax2.clabel(contour_likelihood)

heatmap = ax2.pcolormesh(Hgrid/1000, Omgrid, likelihood)
fig2.colorbar(heatmap)

# %%

# Mrginalised over H_{0} flat in [60, 80]

# Read in generated array for chi^2
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\(60-80) chisq(Om, H0).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 500)
c = 3 * 10**8  # light speed

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_H0 = H0[index[1]]

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# marginalising over H0 - with flat prior:
lik_margin = np.sum(likelihood, axis=1)

# normalise to sum=1
lik_margin /= np.sum(lik_margin)

# set up figure + visuals and plot it:
fig3 = plt.figure()
ax3 = fig3.gca()
ax3.tick_params(labelsize=16)
ax3.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ L(\Omega_{m})$', fontsize=20)
ax3.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax3.plot(Om, lik_margin)

# find peak value and where 68.3% of it lies for 1 \sigma error
Om_found = Om[np.where(lik_margin == np.max(lik_margin))[0]]

variables = st.rv_discrete(values=(Om, lik_margin))
confidence1 = variables.interval(0.683)[1] - Om_found
confidence2 = Om_found - variables.interval(0.683)[0]

print(f'Om = {round(Om_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')

# %%

# Marginalised with gaussian at 73.03 with +- 1.04 as in Riess et al. 2022

# Read in generated array for chi^2
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\(70-76) chisq(Om, H0).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# set up the model axis
H0 = np.linspace(70, 76, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 500)
c = 3 * 10**8  # light speed

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_H0 = H0[index[1]]

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# define Gaussian prior, with approx +- 1.04 as +-1.00 and 73.04 ~ 73.00
g = 1/(np.sqrt(2*np.pi)) * np.exp(-((H0/1000)-73)**2/2)

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# multipying likelihood by Gaussian
weighted = np.multiply(likelihood, g)

# marginalising over H0
lik_margin = np.trapz(weighted, x=H0, axis=1)  # integrating along columns(along H0)

# normalising
lik_margin /= np.sum(lik_margin)

# set up figure + visuals and plot it:
fig5 = plt.figure()
ax5 = fig5.gca()
ax5.tick_params(labelsize=16)
ax5.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ L(\Omega_{m})$', fontsize=20)
ax5.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax5.plot(Om, lik_margin)

# find peak value and where 68.3% of it lies for 1 \sigma error
Om_found = Om[np.where(lik_margin == np.max(lik_margin))[0]]

variables = st.rv_discrete(values=(Om, lik_margin))
confidence1 = variables.interval(0.683)[1] - Om_found
confidence2 = Om_found - variables.interval(0.683)[0]

print(f'Om = {round(Om_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')

