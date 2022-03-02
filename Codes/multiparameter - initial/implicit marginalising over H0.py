'''
Implictly marginalising over different values for H0 using the Hubble free
distance modulus
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

# %%

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

# develop model for each Om, get it's theoretical M and chi^2
i = 0
chisq_array = np.array([])

while i < len(Om):
    # model from list comprehension
    dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl_sum
    
    # interpolate the values to match data size
    dl_model = np.interp(df['z'], z, dl_model)
    
    # define theoretical absolute magnitude from these and use it for model in mu
    M = np.sum((df['mu'] - 5*np.log10(dl_model)) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
    mu_model = 5*np.log10(dl_model) + M - df['mu']
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((mu_model - df['mu'])/(df['dmu']))**2)
    chisq_array = np.append(chisq_array, chisq)

    i += 1


# plot chi^2
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_ylabel(r'$\chi^2$', fontsize=16)
ax.plot(Om, chisq_array)

# plot one of the models
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$z$', fontsize=16)
ax1.set_ylabel(r'$\mu$', fontsize=16)
ax1.plot(np.linspace(0, 1.8, 580), mu_model)


