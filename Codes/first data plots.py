'''
Here we plot the first representation of the data
a plot of mu vs z (distance modulus against redshift)
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

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Distance \ Modulus \ \mu$', fontsize=16)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['mu'], yerr=df['dmu'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# convert the mu data to d_L, make it a new column and sort w.r.t it
df_dL = 10**(0.2*df['mu'] - 5)
df.insert(4, 'dL Mpc', df_dL)
df = df.sort_values('dL Mpc')

# propagate errors and add as new column to data frame
ddL = 0.2*np.log(10)*10**(0.2*df['mu'] - 5) * df['dmu']
df.insert(5, 'ddL Mpc', ddL)

# convert both to Mpc
df['dL Mpc'] = 10**(-6) * df['dL Mpc']
df['ddL Mpc'] = 10**(-6) * df['ddL Mpc']

# plot it with errorbars:
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# plot the data as errorbar plot
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')


# %%
#defining the z-grid
z = np.linspace(0, 1.8, 100)
size = np.size(z)
count = np.linspace(0, np.size(z)-1, np.size(z)) 
count = [int(x) for x in count]

#finding dl using list comprehension
dl_model = [(1+z[j])*z[j]*np.sum(1/(0.23*(1+z[:j])**3+0.77)**0.5)/j for j in count[:]]


#taking H0 = 70km per second per megaparsec
H0 = (70*10**3)
c = 3 * 10**8
#adding in constants here
dl_model = [c* x / H0 for x in dl_model]


plt.plot(z, dl_model)# all off by 10 to the 6 for some reason

#%%repeating the process but using the second method suggested by Tasos 

z = np.linspace(0, 1.8, 100)#defining 100
size = np.size(z)
count = np.linspace(0, np.size(z)-1, np.size(z)) 
count = [int(x) for x in count]
z10 = np.linspace(0, 1.8, 1000)# using 1000 point for sum

dl1_model = [(1+z[j])*z[j]*np.sum(1/(0.23*(1+z10[:10*j])**3+0.77)**0.5)/j for j in count[:]]


H0 = (70*10**3)
c = 3 * 10**8
#adding in constants here
dl1_model = [c* x / H0 for x in dl1_model]








