'''
Attempt including a variety of H_0 in the H(z) model. Produce 2D chi^2
from this, plot it, get the 2D likelihood, plot it. Marginalise over H_0
with flat and gaussian priors. Plot 1D lieklihoods over Om
''' 

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Codes.Module as module
import matplotlib.cm as cm
import scipy.stats as st
import time

# %%

'''
Generating the datafile, runs for some time, so avoid this cell and next one
run proceeding ones that analyse the saved datafile in repo.
'''

start = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('Codes\\Complete Project\\Datasets\\Wei Hz data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = np.linspace(60, 80, 500)

# set up the model axis
Om = np.linspace(0, 0.6, 500)
z = np.linspace(np.min(df['z']), np.max(df['z']), 2000)
zg, Omg = np.meshgrid(z, Om)  # rows are Om and columns are z

# loop over models for all Om
j = 0
chisq_array = np.zeros((len(Om),len(H0)))

while j < len(H0):
    
    # develop models for all z and Om
    model = H0[j] * np.sqrt(Omg * (1+zg)**3 - Omg + 1)
    
    # interpolate the values to match data size
    model_interp = np.zeros(np.shape(np.meshgrid(df['z'], Om)[0]))
    
    for i in range(len(Om)):
        model_interp[i, :] = np.interp(x=df['z'], xp=z, fp=model[i, :])
    
    # get chi^2 value for this Om
    arg = (model_interp - df['Hz'].to_numpy())**2/(df['dHz'].to_numpy())**2
    chisq_array[:, j] = np.sum(arg, axis=1)
    
    # update
    print(f'done {j} out of {len(H0)}')
    j += 1

end = time.perf_counter()
print(f'time to run: {end - start}')

# %%

# to save results: (make sure to change names each time it's run)

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('2D chisquared for H(z) 500.xlsx')

# write DataFrame to excel
dataframe.to_excel(datatoexcel)

# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')

# %%

# Plot chi^2 from saved datafile

# Read in generated array
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\2D chisquared for H(z) 500.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# prepare grids and chi^2
H0 = H0 = np.linspace(60, 80, 500)
Om = np.linspace(0, 0.6, 500)
Hgrid, Omgrid = np.meshgrid(H0, Om)
chisq_array -= np.min(chisq_array)

# figure and cosmetics
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_ylabel(r'$\Omega_{m}$', fontsize=20)
ax.set_xlabel(r'$H_0 \ km \ s^{-1} \ Mpc^{-1}$', fontsize=20)

# plot with colourmap and contours on chi^2 for 2 params
heatmap = ax.pcolormesh(Hgrid, Omgrid, chisq_array)
contourplot = ax.contour(Hgrid, Omgrid, chisq_array, np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
cbar = fig.colorbar(heatmap)
cbar.ax.tick_params(labelsize=16)

# %%

# construct likelihood from it, plot and flat marginalisation


# Read in generated array
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\2D chisquared for H(z) 500.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# prepare grids and chi^2
H0 = np.linspace(60, 80, 500)
Om = np.linspace(0, 0.6, 500)
Hgrid, Omgrid = np.meshgrid(H0, Om)
chisq_array -= np.min(chisq_array)

like = np.exp(-(chisq_array**2)/2)

# find confidences in likelihood with our own function for this:

# normalise it to Volume = 1: with our integrate 2D fucntion:
norm = module.integrate2D(like, Hgrid, Omgrid)
like *= 1/norm

# return for user to inspect
print(f'Our integration of likelihood before normalising: {np.round(norm, 4)}')

# give it to the confidence fucntion to get sigma regions:
heights = module.confidence(like, Hgrid, Omgrid, accu=1000)

# plot as a contour map and heat map
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_ylabel('$\Omega_{m} $', fontsize=20)
ax1.set_xlabel('$H_{0} \ kms^{-1}Mpc^{-1}$', fontsize=20)
contour_likelihood = ax1.contour(Hgrid, Omgrid, like, heights, cmap=cm.jet)

heatmap = ax1.pcolormesh(Hgrid, Omgrid, like)
cbar = fig1.colorbar(heatmap)
cbar.ax.tick_params(labelsize=16)


# #############################################################################
# flat prior marginalising
# #############################################################################

# marginalising over Om - with flat prior:
lik_margin = np.sum(like, axis=1)

# normalise to sum=1 for scip rvdiscrete.
lik_margin /= np.sum(lik_margin)

# set up figure + visuals and plot it:
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=16)
ax2.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ L(\Omega_{m})$', fontsize=20)
ax2.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax2.plot(Om, lik_margin)

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

# Gaussian marginalising around Riess 2022 H0

# Read in generated array
chisq_df = pd.read_excel('Codes\\Complete Project\\Datasets\\2D chisquared for H(z) 500.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# prepare grids and chi^2
H0 = np.linspace(60, 80, 500)
Om = np.linspace(0, 0.6, 500)
Hgrid, Omgrid = np.meshgrid(H0, Om)
chisq_array -= np.min(chisq_array)

like = np.exp((-chisq_array**2)/2)

# #############################################################################
# Gaussian prior marginalising
# #############################################################################

# defining Gaussian
g = 1/(np.sqrt(2*np.pi)) * np.exp(-((H0)-73)**2/2)

# multipying likelihood by Gaussian
weighted = np.multiply(like, g)

# marginalising over H0
lik_margin = np.trapz(weighted, x=H0, axis=1)  # integrating along columns(along H0)

# normalising
lik_margin /= np.sum(lik_margin)

# find peak value and where 68.3% of it lies for 1 \sigma error
Om_found = Om[np.where(lik_margin == np.max(lik_margin))[0]]

variables = st.rv_discrete(values=(Om, lik_margin))
confidence1 = variables.interval(0.683)[1] - Om_found
confidence2 = Om_found - variables.interval(0.683)[0]

# set up figure + visuals and plot it:
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=16)
ax2.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ L(\Omega_{m})$', fontsize=20)
ax2.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax2.plot(Om, lik_margin)

print(f'Om = {round(Om_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')
