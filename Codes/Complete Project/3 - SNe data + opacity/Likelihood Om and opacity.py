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

# Likelihood and contour plots with our confidence finder

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
eps = np.linspace(-0.3, 0.3, 500)
z = np.linspace(np.min(df['z']), 1.8, 100)

# finding minimum of chisquared coords
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_eps = eps[index[1]]

# set up plotting axis
epsgrid, Omgrid = np.meshgrid(eps, Om)

# switch to likelihoods
likelihood = np.exp(-(chisq_array**2)/2)

# find confidences in likelihood with our own function for this:

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(likelihood, epsgrid, Omgrid, interp=1000)
likelihood *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(likelihood, epsgrid, Omgrid, accu=10000, interp=1000)

# plot as a contour map and heat map
fig4 = plt.figure()
ax4 = fig4.gca()
ax4.tick_params(labelsize=16)
ax4.set_ylabel('$\Omega_{m} $', fontsize=20)
ax4.set_xlabel('$\epsilon$', fontsize=20)
contour_likelihood = ax4.contour(epsgrid, Omgrid, likelihood, heights, cmap=cm.jet)
#ax4.clabel(contour_likelihood)

heatmap = ax4.pcolormesh(epsgrid, Omgrid, likelihood)
fig4.colorbar(heatmap)

# %%

# marginalise over Om in likelihood
# flat prior in Om

# Read in generated array for chi^2
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\chisq(Om, eps) (500 points).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# set up the model axis
Om = np.linspace(0, 0.6, 500)
eps = np.linspace(-0.3, 0.3, 500)
z = np.linspace(np.min(df['z']), 1.8, 100)
c = 3 * 10**8  # light speed

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_eps = eps[index[1]]

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# marginalising over Om - with flat prior:
lik_margin = np.sum(likelihood, axis=0)

# normalise to sum=1
lik_margin /= np.sum(lik_margin)

# set up figure + visuals and plot it:
fig5 = plt.figure()
ax5 = fig5.gca()
ax5.tick_params(labelsize=16)
ax5.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ and \ \Omega_{m} \ L(\epsilon)$', fontsize=20)
ax5.set_xlabel(r'$\epsilon$', fontsize=20)
ax5.plot(eps, lik_margin)

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

# %%

# marginalise over Om in likelihood
# Gaussian prior

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
eps = np.linspace(-1, 1, 200)
Om = np.linspace(0, 1, 500)
z = np.linspace(np.min(df['z']), 1.8, 100)

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0

# defining Gaussian
g = np.exp(-(Om - 0.23)**2/(2* 0.05**2))/(0.05*np.sqrt(2*np.pi))

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

# multipying likelihood by Gaussian
#weighted = likelihood * g  # need to multiply all rows for set column by all values of g


# slow method for now, easily improved soon!
weighted = np.zeros(np.shape(likelihood))
i = 0
while i < len(g):
    weighted[i, :] = likelihood[i, :]*g[i]
    i += 1

# marginalising over Om
lik_margin = np.trapz(weighted, x=Om, axis=0)  # integrating along rows (Om)

# normalising
lik_margin /= np.sum(lik_margin)

# find peak value and where 68.3% of it lies for 1 \sigma error
eps_found = eps[np.where(lik_margin == np.max(lik_margin))[0]]
variables = st.rv_discrete(values=(eps, lik_margin))
confidence1 = variables.interval(0.683)[1] - eps_found
confidence2 = eps_found - variables.interval(0.683)[0]

# plot result:
fig = plt.figure()
ax = fig.gca()
ax.plot(eps, lik_margin)
ax.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ and \ \Omega_{m} \ L(\epsilon)$', fontsize=16)
ax.set_xlabel(r'$\epsilon$', fontsize=16)

# return values
print(f'eps = {round(eps_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')

