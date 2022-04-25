
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import Codes.Module as module

# %%

# Read in Sne data as pandas dataframe
df = pd.read_excel('Codes\\Complete Project\\Datasets\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 73*10**3  # unimportant value here, marginalised over anyway
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 500)
z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z1000 = np.linspace(0, z, 1000)  # inetrgal approximation axis

# develop models for each Om, get it's theoretical M and chi^2
i = 0
chisq_array = np.array([])
models_mu = np.zeros((len(df['z']), len(Om)))

while i < len(Om):
    # model from list comprehension
    combs = 1/np.sqrt(Om[i]*(1+z1000)**3 - Om[i] + 1)
    dl_sum = np.sum(combs, axis=0)
    dl_model = (c/H0)*(1+z)*z/1000 * dl_sum
    
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
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax.set_ylabel(r'$\chi^2$', fontsize=20)
ax.plot(Om, chisq_array)

# plot model in mu with minium chi^2
index_min = np.where(chisq_array == np.min(chisq_array))[0][0]
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=16)
ax1.set_xlabel(r'$z$', fontsize=20)
ax1.set_ylabel(r'$\mu$', fontsize=20)
for i in [100, 150, 200, 250, 299]:
    ax1.plot(df['z'], models_mu[:, i], label=fr'$Model \ \Omega_m ={round(Om[i], 4)}$')  # interpolated z
ax1.plot(df['z'], models_mu[:, index_min], label=rf'$minimum \ \chi^2 \ model \ \Omega_m ={round(Om[index_min], 4)}$')

ax1.legend()

# define relevant chi^2 regions and plot there only with confidence:
Delta_squared = 20
print(f'minimum chi^2 at {np.min(chisq_array)}')
chisq_array = chisq_array - np.min(chisq_array)
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

# plot reduced chi^2
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=16)
ax2.set_xlabel(r'$\Omega_{m}$', fontsize=20)
ax2.set_ylabel(r'$\chi^2$', fontsize=20)
ax2.set_ylim(0, 20)
ax2.plot(Om, chisq_array, 'k')

# add plot of confidence regions:
lerror, rerror = module.chi_confidence1D(chisq_array, Om, ax2)

print('\n')
print('minimum value was found to be at Omega_m = ')
print(Om[np.where(chisq_array == np.min(chisq_array))[0][0]])
print('\n')
print('the starndard deviation was found to be:')
print(f' + {np.round(lerror, 5)}')
print(f' - {np.round(rerror, 5)}')

# %%

# Checking M_th

# Read in Sne data as pandas dataframe
df = pd.read_excel('Codes\\Complete Project\\Datasets\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 73*10**3  # unimportant value here, marginalised over anyway
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 500)
#z = np.linspace(0, 1.8, 100)
z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z1000 = np.linspace(0, z, 1000)  # inetrgal approximation axis

# develop models for each Om, get it's theoretical M and chi^2
i = 0
chisq_array = np.array([])
models_mu = np.zeros((len(df['z']), len(Om)))
Mth = []

while i < len(Om):
    # model from list comprehension
    combs = [1/np.sqrt(Om[i]*(1+z1000[:,j])**3 - Om[i] + 1) for j in count[:]]
    dl_sum = np.sum(combs, axis=1)
    dl_model = (c/H0)*(1+z)*z/1000 * dl_sum
    
    #dl_model[0] = dl_model[1] - (dl_model[2] - dl_model[1])
    
    # interpolate the values to match data size
    dl_model_interp = np.interp(x=df['z'], xp=z, fp=dl_model)
    
    # define theoretical absolute magnitude from these and use it for model in mu
    M = np.sum((df['mu'] - 5*np.log10(dl_model_interp)) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
    mu_model_interp = 5*np.log10(dl_model_interp) + M
    Mth.append(M)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((mu_model_interp - df['mu'])**2/(df['dmu'])**2))
    chisq_array = np.append(chisq_array, chisq)
    
    models_mu[:, i] = mu_model_interp

    i += 1


# define relevant chi^2 regions and plot there only with confidence:
Delta_squared = 20
chisq_array = chisq_array - np.min(chisq_array)
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
ax1.set_ylim(0, 20)
ax1.plot(Om, chisq_array)


# add plot of confidence regions:
lerror, rerror = module.chi_confidence1D(chisq_array, Om, ax1)

print('\n')
print('minimum value was found to be at Omega_m = ')
print(Om[np.where(chisq_array == np.min(chisq_array))[0][0]])
print('\n')
print('the starndard deviation was found to be:')
print(f' + {np.round(lerror, 5)}')
print(f' - {np.round(rerror, 5)}')

# get variation in M_th:
Mth_mean = np.mean(Mth)
Mth_std = np.std(Mth)

print('Mean value of Mth was:')
print(f'                       {Mth_mean}')
print('\n')
print('With standard deviation:')
print(f'                       {Mth_std}')

