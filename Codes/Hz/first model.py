'''
We develop our first model for H(z) with zero curvature.
We minimise chi^2 for it w.r.t H(z) data
and plot that on the data.
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Codes.Module as module
import matplotlib.cm as cm
import time

# %%

start = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\Hz\\Wei Hz data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70

# set up the model axis
Om = np.linspace(0, 0.6, 1000)
z = np.linspace(np.min(df['z']), np.max(df['z']), 2000)

# loop over models for all Om
i = 0
chisq_array = np.zeros(np.shape(Om))

while i < len(Om):
    # develop model from equation as for E(z)
    model = H0 * np.sqrt(Om[i]*(1+z)**3 - Om[i] + 1)
    
    # interpolate the values to match data size
    model_interp = np.interp(x=df['z'], xp=z, fp=model)
    
    # get chi^2 value for this Om
    chisq = np.sum(((model_interp - df['Hz'])/(df['dHz']))**2)
    
    # save to array and update index
    chisq_array[i] = chisq
    i += 1


end = time.perf_counter()
print(f'time to run: {end - start}')


# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]


# plot chi^2 reduced to wanted range
Delta_squared = 20
print(f'minimum of chi^2 at : {round(np.min(chisq_array), 1)}')
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=18)
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax1.set_ylabel(r'$\chi^2$', fontsize=20)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_array)

reg1, reg2 = module.chi_confidence1D(chisq_array, Om, ax1)  # plot confidence regions

# print confidences to user to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)}')
print('1-sigma error =')
print(f'               + {np.round(reg2, 5)}')
print(f'               - {np.round(reg1, 5)}')
print('\n')



# %%

# Same, Semi Vectorised

start = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\Hz\\Wei Hz data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70

# set up the model axis and 2D grids
Om = np.linspace(0, 0.6, 1000)
z = np.linspace(np.min(df['z']), np.max(df['z']), 2000)
zg, Omg = np.meshgrid(z, Om)  # rows are Om and columns are z

# develop models for all z and Om
model = H0 * np.sqrt(Omg * (1+zg)**3 - Omg + 1)

# interpolate the values to match data size
model_interp = np.zeros(np.shape(np.meshgrid(df['z'], Om)[0]))
for i in range(len(Om)):
    model_interp[i, :] = np.interp(x=df['z'], xp=z, fp=model[i, :])

# get chi^2 value for this Om
arg = (model_interp - df['Hz'].to_numpy())**2/(df['dHz'].to_numpy())**2
chisq_array = np.sum(arg, axis=1)

end = time.perf_counter()
print(f'time to run: {end - start}')

# %%
# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

# plot full chi^2
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=18)
ax.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax.set_ylabel(r'$\chi^2$', fontsize=20)

ax.plot(Om, chisq_array)

# plot chi^2 reduced to wanted range
Delta_squared = 20
print(f'minimum of chi^2 at : {round(np.min(chisq_array), 1)}')
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig1 = plt.figure()
ax1 = fig1.gca()
ax1.tick_params(labelsize=18)
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax1.set_ylabel(r'$\chi^2$', fontsize=20)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_array)

reg1, reg2 = module.chi_confidence1D(chisq_array, Om, ax1)  # plot confidence regions

# print confidences to user to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)}')
print('1-sigma error =')
print(f'               + {np.round(reg2, 5)}')
print(f'               - {np.round(reg1, 5)}')
print('\n')

# plot minimum chi^2 model on the data.
model = H0 * np.sqrt(min_Om*(1+z)**3 - min_Om + 1)
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.tick_params(labelsize=18)
ax2.set_xlabel(r'$z$', fontsize=20)
ax2.set_ylabel(r'$H(z) \ kms^{-1} Mpc^{-1}$', fontsize=20)

ax2.errorbar(df['z'], df['Hz'], yerr=df['dHz'],
            capsize=2, fmt='.', markersize=5, ecolor='k')
ax2.plot(z, model)

