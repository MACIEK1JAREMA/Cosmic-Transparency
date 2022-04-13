'''
Here, we approximate the derived integral for the general luminosty distance
for a flat universe with \Omega_m = 0.23 and plot the results
as a function of d_L over z
takes H_0 as 70km s^-1 Mpc^-1
'''

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

# define the z axis
z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)+1

#count = [int(x) for x in count]  # same thing, much slower

# finding dl using list comprehension
dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z[:j])**3 + OL)) for j in count[:]]
dl_model = (1+z)*z/(count) * dl_sum

# set up figure and plot
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)
plt.title("Basic calculation")
plt.plot(z, dl_model)

# %%

# More accurately

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# axis
z = np.linspace(np.min(df['z']), 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z1000 = np.linspace(0, z, 1000)  # using 1000 point for sum



# model from list comprehension again
combs = [1/np.sqrt(Om*(1+z1000[:,j])**3 - Om + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0)*(1+z)*z/1000 * dl1_sum
# set up figure and plot
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)
plt.plot(z, dl1_model)

# %%

# Plotting with data together:

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
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Calculate models:

# set up axis
z = np.linspace(np.min(df['z']), 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)+1


# Less accurate:
z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)

dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z[:j+1])**3 + OL)) for j in count[:]]
dl_model = (1+z)*z/(count) * dl_sum

# more accurate
z1000 = np.linspace(0, z, 1000)  # using 1000 point for sum
combs = [1/np.sqrt(Om*(1+z1000[:,j])**3 - Om + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0)*(1+z)*z/1000 * dl1_sum

# plot above models, with a legend:
ax.plot(z, dl_model, 'r-', label='approximate')
ax.plot(z, dl1_model, 'g-', label='more accurate')
ax.legend()

# %%

# repeating the process but using mu

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
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=16)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Calculate models:

# set up axis
z = np.linspace(0, 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# Less accurate:

dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z[:j+1])**3 + OL)) for j in count[:]]
dl_model = (1+z)*z/(count) * dl_sum
dlmu_model = 5*np.log10(dl_model) + 25

# more accurate

z1000 = np.linspace(0, z, 1000)  # using 1000 point for sum
combs = [1/np.sqrt(Om*(1+z1000[:,j])**3 - Om + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0)*(1+z)*z/1000 * dl1_sum
dl1mu_model = 5*np.log10(dl1_model) + 25

# plot above models, with a legend:
ax.plot(z, dlmu_model, 'r-', label='approximate')
ax.plot(z, dl1mu_model, 'g-', label='more accurate')
ax.legend()
