'''
introduction of opacity as a parameter in chisquared while implicitly
marginalising over H0
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import time

# %%

start_t = time.perf_counter()


# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 73*10**3  # unimportant value here, marginalised over anyway
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0.1, 0.4, 500)
epsil = np.linspace(-0.2, 0.2, 500)

z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z1000 = np.linspace(0, z, 1000)  # inetrgal approximation axis

# develop models for each Om, get it's theoretical M and chi^2
models_mu = np.zeros((len(df['z']), len(Om)))
chisq_array = np.zeros(np.shape(np.meshgrid(epsil, Om)[0]))

k = 0

while k < len(epsil):
    # define opacity parameter for our desired z values
    tor = 2*epsil[k]*df['z']
    i = 0
    while i < len(Om):
        # model from list comprehension
        combs = [1/np.sqrt(Om[i]*(1+z1000[:, j])**3 - Om[i] + 1) for j in count[:]]
        dl_sum = np.sum(combs, axis=1)
        dl_model = (c/H0)*(1+z)*z/1000 * dl_sum
        
        # interpolate the values to match data size
        dl_model_interp = np.interp(x=df['z'], xp=z, fp=dl_model)
        
        # define theoretical absolute magnitude from these and use it for model in mu
        M = np.sum((df['mu'] - 5*np.log10(dl_model_interp)-2.5*tor*np.log10(np.exp(1))) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
        mu_model_interp = 5*np.log10(dl_model_interp)-2.5*tor*np.log10(np.exp(1)) + M
        # get chi^2 value for this Om and save to its array
        chisq = np.sum(((mu_model_interp - df['mu'])**2/(df['dmu'])**2))
        chisq_array[i, k] = chisq
        
        models_mu[:, i] = mu_model_interp
    
        i += 1
    
    k += 1
    
    print(f'Done k = {k} out of 500')


end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

# to save results: (make sure to change names each time it's run)

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('chisq with opacity and reduced range (500 points).xlsx')

# write DataFrame to excel
dataframe.to_excel(datatoexcel)

# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')