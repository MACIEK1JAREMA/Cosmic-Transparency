'''
We take a first look at likelihood functions, defining a 1D L, with Om only
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import time

# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 500)
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
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array = np.append(chisq_array, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

# #############################################################################
# crop to relevant bounds and get likelihood + plot
# #############################################################################

# define relevant delta chi^2 region, more than 3 sigma (at Delta_chisqraued~9)
Delta_squared = 20

# reduce chi^2 and corr. Om
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

# define likelihood from these:
likelihood = np.exp((-chisq_array**2)/2)

# normalise to sum=1
likelihood /= np.sum(likelihood)

fig5 = plt.figure()
ax5 = fig5.gca()
ax5.set_ylabel(r'$Likelihood \  L(\Omega_{m})$', fontsize=16)
ax5.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax5.plot(Om, likelihood)


# find peak value and where 68.3% of it lies for 1 \sigma error
Om_found = Om[np.where(likelihood == np.max(likelihood))[0]]

variables = st.rv_discrete(values=(Om, likelihood))
confidence1 = variables.interval(0.683)[1] - Om_found
confidence2 = Om_found - variables.interval(0.683)[0]

print(f'Om = {round(Om_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')
