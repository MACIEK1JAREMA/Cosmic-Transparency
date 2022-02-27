'''
Here we look at including H0 as a parameter, then conpute a 2D
likelihood funciton to marginalise over H0 variation and constrain \Omega_m
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(65, 75, 300)
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

i = 0
j = 0
chisq_array = np.zeros(np.shape(np.meshgrid(H0, Om)[0]))

while i < len(H0):
    j = 0
    while j < len(Om):
        # model from list comprehension
        dl1_sum = [(c/H0[i]) * np.sum(1/np.sqrt(Om[j]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[j] + 1)) for j in count[:]]
        dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
        # convert to mu vs z to compare to data.
        dl1mu_model = 5*np.log10(dl1_model) + 25
        
        # interpolate the values in the grid as they are generated
        interp = np.interp(df['z'], z, dl1mu_model)
        
        # get chi^2 value for this Om and save to its array
        chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
        chisq_array[i, j] = chisq
        
        j += 1
    i += 1

# %%
# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_array, label='$\chi^2 \ of \ model \ with \ H_0=70 \ km s^{-1} Mpc^{-1}$', color='k')

# #############################################################################
# find corresponding Om at confidence region boundaries
# #############################################################################

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_array)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)}')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om, 5)}')
print(f'               - {round(min_Om - Omi[indx1][0][0], 5)}')
print('\n')

# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')
