'''
Here, we further investigate the effect of our found error in calculating
chi^2 that could affect all of our prior work
'''

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
z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = np.linspace(0, len(z10)-1, len(z10)).astype(int) + 1

i = 0
chisq_array = np.array([])

while i < len(Om):
    # model from list comprehension
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # delete the first uninteresting -inf at dL=0 by extrapolating 1 point linearly
    #dl1mu_model[0] = dl1mu_model[1] - (dl1mu_model[2] - dl1mu_model[1])
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # plot one interpolated model example
    if i == 100:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel(r'$z$', fontsize=20)
        ax.set_ylabel(r'$\mu$', fontsize=20)
        ax.plot(df['z'], interp, lw=5, label=r'Interpolated')
        ax.plot(z, dl1mu_model, label=r'original')
        ax.legend(fontsize=18)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array = np.append(chisq_array, chisq)

    i += 1

# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

print(f'min Om = {np.round(min_Om, 5)}')

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)

ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_array, label='$\chi^2 \ of \ model \ with \ H_0=70 \ km s^{-1} Mpc^{-1}$', color='k')

# plot confidence regions
lower, upper = module.chi_confidence1D(chisq_array, Om, ax1)

print(f'Error was: + {np.round(upper[0] ,5)}')
print(f'           - {np.round(lower[0] ,5)}')

# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

# %%

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 73*10**3  # unimportant value here, marginalised over anyway
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)
#z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = np.array((np.linspace(0, len(z10)-1, len(z10)).astype(int))) +1

# develop models for each Om, get it's theoretical M and chi^2
i = 0
chisq_array = np.array([])
models_mu = np.zeros((len(df['z']), len(Om)))

while i < len(Om):
    # model from list comprehension
    dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl_sum
    
    #dl_model[0] = dl_model[1] - (dl_model[2] - dl_model[1])
    
    # interpolate the values to match data size
    dl_model_interp = np.interp(x=df['z'], xp=z, fp=dl_model)
    
    # define theoretical absolute magnitude from these and use it for model in mu
    M = np.sum((df['mu'] - 5*np.log10(dl_model_interp)) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
    mu_model_interp = 5*np.log10(dl_model_interp) + M
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((mu_model_interp - df['mu'])**2/(df['dmu'])**2))
    chisq_array = np.append(chisq_array, chisq)
    
    models_mu[:, i] = mu_model_interp

    i += 1


# plot chi^2 initially
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_ylabel(r'$\chi^2$', fontsize=16)
ax.plot(Om, chisq_array)

# plot model in mu with minium chi^2
index_min = np.where(chisq_array == np.min(chisq_array))[0][0]
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$z$', fontsize=16)
ax1.set_ylabel(r'$\mu$', fontsize=16)
for i in [100, 150, 200, 250, 299]:
    ax1.plot(df['z'], models_mu[:, i], label=fr'$Model \ \Omega_m ={round(Om[i], 4)}$')  # interpolated z
ax1.plot(df['z'], models_mu[:, index_min], label=rf'$minimum \ \chi^2 \ model \ \Omega_m ={round(Om[index_min], 4)}$')

ax1.legend()

# define relevant chi^2 regions and plot there only with confidence:
Delta_squared = 20
chisq_array = chisq_array - np.min(chisq_array)
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig2 = plt.figure()
ax2 = fig2.gca()
ax2.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax2.set_ylabel(r'$\chi^2$', fontsize=16)
ax2.set_ylim(0, 20)
ax2.plot(Om, chisq_array)

# plot confidence regions
lower, upper = module.chi_confidence1D(chisq_array, Om, ax2)

print('\n')
print('minimum value was found to be at Omega_m = ')
print(Om[np.where(chisq_array == np.min(chisq_array))[0][0]])
print('\n')
print('the starndard deviation was found to be:')
print(f' + {np.round(upper, 5)}')
print(f' - {np.round(lower, 5)}')
