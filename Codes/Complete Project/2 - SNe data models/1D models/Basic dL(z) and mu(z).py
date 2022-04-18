
# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

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
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$Redshift \ z$', fontsize=20)
ax.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=20)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# redshift axis
z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# inetrgal approximation axis (z')
z1000 = np.linspace(0, z, 1000)

int_arg = 1/np.sqrt(Om*(1+z1000)**3 + 1 - Om)
dl_sum = np.sum(int_arg, axis=0)
dl_model = (c/H0)*(1+z)*(z/1000) * dl_sum

# model from list comprehension
#combs = [1/np.sqrt(Om*(1+z1000[:, j])**3 - Om + 1) for j in count[:]]
#dl_sum = np.sum(combs, axis=1)
#dl_model = (c/H0)*(1+z)*z/1000 * dl_sum

# plot model
plt.plot(z, dl_model)

# %%

# repeating the process but using mu

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$Redshift \ z$', fontsize=20)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=20)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# redshift axis
z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# inetrgal approximation axis (z')
z1000 = np.linspace(0, z, 1000)

int_arg = 1/np.sqrt(Om*(1+z1000)**3 + 1 - Om)
dl_sum = np.sum(int_arg, axis=0)
dl_model = (c/H0)*(1+z)*(z/1000) * dl_sum
mu_model = 5*np.log10(dl_model) + 25

# plot model
plt.plot(z, mu_model)

# %%

# Same but with residuals

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# set up figure and visuals
fig = plt.figure()
ax = fig.add_axes((0.05, 0.30, 0.90, 0.65))
ax.tick_params(labelsize=16)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=20)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=3, ecolor='k')

# redshift axis
z = np.linspace(np.min(df['z']), 1.5, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# inetrgal approximation axis (z')
z1000 = np.linspace(0, z, 1000)

int_arg = 1/np.sqrt(Om*(1+z1000)**3 + 1 - Om)
dl_sum = np.sum(int_arg, axis=0)
dl_model = (c/H0)*(1+z)*(z/1000) * dl_sum
mu_model = 5*np.log10(dl_model) + 25

# plot model
plt.plot(z, mu_model)

# Plot residuals

# interpolate
mu_interp = np.interp(df['z'], z, mu_model)

# set up axis for residuals
box = fig.add_axes((0.05, 0.10, 0.90, 0.15))

box.set_xlabel(r'$Redshift \ z$', fontsize=20)

box.errorbar(df['z'], df['mu'] - mu_interp, yerr=df['dmu'],
            capsize=2, fmt='.', markersize=3, ecolor='k')

box.axhline(0, color='r', linestyle='--')





