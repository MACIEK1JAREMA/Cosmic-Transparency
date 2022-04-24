
# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import Codes.Module as module

# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, z, 1000)  # inetrgal approximation axis z'

i = 0  # loop counter
chisq_array = np.array([])  # array to store chi^2 results

while i < len(Om):
    # model for d_L
    int_arg = 1/np.sqrt(Om[i]*(1+z10)**3 + 1 - Om[i])  # integrand
    dl_sum = np.sum(int_arg, axis=0)  # integral result
    dl_model = (c/H0)*(1+z)*(z/len(z10)) * dl_sum  # model
    
    # convert to mu vs z to compare to data.
    dlmu_model = 5*np.log10(dl_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dlmu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array = np.append(chisq_array, chisq)
    
    i += 1  # increment Om


# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

print(f'found minimum chi^2 as: {min_chisq}')

# set up figure for full chi^2 and plot
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax.set_ylabel(r'$\chi^2$', fontsize=20)
ax.plot(Om, chisq_array)

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

# set up figure for reduced chi^2:
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax1.set_ylabel(r'$\chi^2$', fontsize=20)
ax1.set_ylim(0, 20)
# plot it:
ax1.plot(Om, chisq_array, label='$\chi^2 \ of \ model \ with \ H_0=70 \ km s^{-1} Mpc^{-1}$', color='k')

# #############################################################################
# find corresponding Om at confidence region boundaries
# #############################################################################

# plot confidence regions
error_left, error_right = module.chi_confidence1D(chisq_array, Om, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)}')
print('1-sigma error =')
print(f'               + {round(error_right[0], 5)}')
print(f'               - {round(error_left[0], 5)}')
print('\n')

# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

# %%

# develop the model that corresponds to the found Om, and plot it on data
# also plot the one when using Om = 0.23

# Plotting with data together:

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# convert the mu data to d_L, make it a new column and sort w.r.t it
df_dL = 10**(0.2*df['mu'] - 5)
df.insert(4, 'dL Mpc', df_dL)
df = df.sort_values('dL Mpc')

# propagate errors and add as new column to data frame
ddL = 0.2*np.log(10)*10**(0.2*df['mu'] - 5) * df['dmu']
df.insert(5, 'ddL Mpc', ddL)

# set up figure and visuals
fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# plot the data as errorbar plot
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Calculate models:

# set up axis
z = np.linspace(np.min(df['z']), 1.8, 100)  # for model
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, 1.8, 1000)  # for integral approximation
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))
# develop model
combs = [1/np.sqrt(min_Om*(1+z1000[:,j])**3 - min_Om + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0)*(1+z)*z/1000 * dl1_sum

# plot above models, with a legend:
ax1.plot(z, dl1_model, 'g-', label=rf'$Model \ with \ \chi^2 \ minimised \ \Omega_m \ = \ {round(min_Om, 4)}$')


# plot with Om = 0.23
combs = [1/np.sqrt(0.23*(1+z1000[:,j])**3 +0.77) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0)*(1+z)*z/1000 * dl1_sum

# plot above models, with a legend:
ax1.plot(z, dl1_model, 'r-', label=rf'$Model \ with \ \Omega_m \ = \ 0.23$')

ax1.legend()
