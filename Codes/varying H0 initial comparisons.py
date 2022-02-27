'''
Here we look at the effect that a changed value of H0 has on the end models
we do so on the example of k=0 models with chi^2 analysis
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%

# Define constants
H0_lambdacdm = 67*10**3  # H0 approximately as found by distance ladder, predicted by LCDM
H0_SHOES = 73*10**3  # WMAP H0
H0_mean = 70*10**3
c = 3 * 10**8  # light speed

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

# Set up the plot for the model and data + visuals
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)
# plot data
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Set up a plot for chi^2 analysis + visuals
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax2.set_ylabel(r'$\chi^2$', fontsize=16)

# set up the model axis
Om = np.linspace(0, 1, 100)
z = np.linspace(0, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

# ############################################################################
# complete the model for LCDM
# ############################################################################

i = 0
chisq_array_LCDM = np.array([])

while i < len(Om):
    # model from list comprehension
    dl1_sum = [(c/H0_lambdacdm) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_LCDM = np.append(chisq_array_LCDM, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_LCDM.argmin()
min_Om_LCDM = Om[index]
min_chisq_LCDM = chisq_array_LCDM[index]

# develop the model on this found Om
dl1_sum = [(c/H0_lambdacdm) * np.sum(1/np.sqrt(min_Om_LCDM*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - min_Om_LCDM + 1)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot chi^2 against Omega_m for it
ax1.plot(z, dl1_model, 'b-', label=rf'$H_0 \ = \ 67 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_LCDM, 4)}$')
ax2.plot(Om, chisq_array_LCDM, 'b-', label = '$H_0 \ = \ 67 \ km \ s^{-1} \ Mpc^{-1}$')

print('minimising \chi^2 for H0=67 kms-1Mpc-1 gives a matter density of ', min_Om_LCDM)

# ############################################################################
# Repeat for other H0
# ############################################################################

i = 0
chisq_array_WMAP = np.array([])

while i < len(Om):
    # model from list comprehension
    dl1_sum = [(c/H0_SHOES) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_WMAP = np.append(chisq_array_WMAP, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_WMAP.argmin()
min_Om_WMAP = Om[index]
min_chisq_WMAP = chisq_array_WMAP[index]

# develop the model on this found Om
dl1_sum = [(c/H0_lambdacdm) * np.sum(1/np.sqrt(min_Om_WMAP*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - min_Om_WMAP + 1)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot chi^2 against Omega_m
ax1.plot(z, dl1_model, 'r-', label=rf'$H_0 \ = \ 73 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_WMAP, 4)}$')
ax2.plot(Om, chisq_array_WMAP, 'r-', label = '$H_0 \ = \ 73 \ km \ s^{-1} \ Mpc^{-1}$')

print('minimising \chi^2 gives a matter density of ', min_Om_WMAP)

# ############################################################################
# repeat for the mean, as used before, to compare.
# ############################################################################

i = 0
chisq_array_mean = np.array([])

while i < len(Om):
    # model from list comprehension
    dl1_sum = [(c/H0_mean) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_mean = np.append(chisq_array_mean, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_mean.argmin()
min_Om_mean = Om[index]
min_chisq_mean = chisq_array_WMAP[index]

# develop the model on this found Om
dl1_sum = [(c/H0_mean) * np.sum(1/np.sqrt(min_Om_mean*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - min_Om_mean + 1)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot chi^2 against Omega_m
ax1.plot(z, dl1_model, 'k-', label=rf'$H_0 \ = \ 70 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_mean, 4)}$')
ax2.plot(Om, chisq_array_mean, 'k-', label = '$H_0 \ = \ 70 \ km \ s^{-1} \ Mpc^{-1}$')

# finilise plot legends
ax2.legend()
ax1.legend()

