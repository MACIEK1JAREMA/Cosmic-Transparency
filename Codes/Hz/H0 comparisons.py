'''
Produce a sinlge plot of \chi^2s for a variety of H_0
Compare chi^2 minima and their locations along Om axis
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Codes.Module as module

# %%

# Read in data as pandas dataframe
df = pd.read_excel('data\\Hz\\Wei Hz data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = np.array([67, 70, 73])

# set up the model axis
Om = np.linspace(0, 0.6, 1000)
z = np.linspace(np.min(df['z']), np.max(df['z']), 2000)
zg, Omg = np.meshgrid(z, Om)  # rows are Om and columns are z

# setting up axes
fig = plt.figure()  # plot full chi^2
ax = fig.gca()
ax.tick_params(labelsize=18)
ax.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax.set_ylabel(r'$\chi^2$', fontsize=20)

fig1 = plt.figure()  # chi^2 in reduced range
ax1 = fig1.gca()
ax1.tick_params(labelsize=18)
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=20)
ax1.set_ylabel(r'$\chi^2$', fontsize=20)
ax1.set_ylim(0, 20)

fig2 = plt.figure()  # models on data
ax2 = fig2.gca()
ax2.tick_params(labelsize=18)
ax2.set_xlabel(r'$z$', fontsize=20)
ax2.set_ylabel(r'$H(z) \ kms^{-1} Mpc^{-1}$', fontsize=20)
ax2.errorbar(df['z'], df['Hz'], yerr=df['dHz'],
             capsize=2, fmt='.', markersize=5, ecolor='k')

# loop over models for each H_0
j = 0
while j < len(H0):
    # reset parameters
    i = 0
    Om = np.linspace(0, 0.6, 1000)
    chisq_array = np.zeros(np.shape(Om))
    
    # develop models for all z and Om
    model = H0[j] * np.sqrt(Omg * (1+zg)**3 - Omg + 1)
    
    # interpolate the values to match data size
    model_interp = np.zeros(np.shape(np.meshgrid(df['z'], Om)[0]))
    for i in range(len(Om)):
        model_interp[i, :] = np.interp(x=df['z'], xp=z, fp=model[i, :])
    
    # get chi^2 value for this Om
    arg = (model_interp - df['Hz'].to_numpy())**2/(df['dHz'].to_numpy())**2
    chisq_array = np.sum(arg, axis=1)

    # find minima
    index = chisq_array.argmin()
    min_Om = Om[index]
    min_chisq = chisq_array[index]
    
    # plot full chi^2
    ax.plot(Om, chisq_array, label=rf'$H_0 \ = \ {round(H0[j], 4)}\ km \ s^{-1} \ Mpc^{-1}$')
    
    # plot chi^2 reduced to wanted range
    Delta_squared = 20
    print(f'for H_{0}={H0[j]}, minimum of chi^2 at : {round(np.min(chisq_array), 1)}')
    chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
    in_index = np.where(chisq_array <= Delta_squared)
    chisq_array = chisq_array[in_index]  # only keep in wanted region
    Om = Om[in_index]  # crop Om accordingly
    
    ax1.plot(Om, chisq_array, label = rf'$H_0 \ = \ {round(H0[j], 4)}\ km \ s^{-1} \ Mpc^{-1}$')
    
    # plot confidence regions
    if j == len(H0)-1:
        reg1, reg2 = module.chi_confidence1D(chisq_array, Om, ax1)
    else:
        reg1, reg2 = module.chi_confidence1D(chisq_array, Om, ax1, labels=False) # no label repeats
    
    # print confidences to user to user:
    print('\n')
    print(f'minimising chi^2 gives a matter density = {round(min_Om, 4)}')
    print('1-sigma error =')
    print(f'               + {np.round(reg2, 5)}')
    print(f'               - {np.round(reg1, 5)}')
    print('\n')
    
    model = H0[j] * np.sqrt(min_Om*(1+z)**3 - min_Om + 1)
    
    # plot resulting model for current H_0
    ax2.plot(z, model, label = rf'$H_0 \ = \ {round(H0[j], 1)}\ km \ s^{-1} \ Mpc^{-1}$')
    
    # finilise legends
    ax.legend()
    ax1.legend(loc=1)
    ax2.legend()
    
    j += 1
