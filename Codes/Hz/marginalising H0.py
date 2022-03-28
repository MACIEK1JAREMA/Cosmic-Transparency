'''
Attempt including a variety of H_0 in the H(z) model. Produce 2D chi^2
from this, plot it, get the 2D likelihood, plot it. Marginalise over H_0
with flat and gaussian priors. Plot 1D lieklihoods over Om
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
H0 = np.linspace(60,80, 500)

# set up the model axis
Om = np.linspace(0, 0.6, 1000)
z = np.linspace(np.min(df['z']), np.max(df['z']), 2000)

# loop over models for all Om
j = 0
chisq_array = np.zeros((len(Om),len(H0)))
while j < len(H0):
    i=0
    while i < len(Om):
        # develop model from equation as for E(z)
        model = H0[j] * np.sqrt(Om[i]*(1+z)**3 - Om[i] + 1)
        
        # interpolate the values to match data size
        model_interp = np.interp(x=df['z'], xp=z, fp=model)
        
        # get chi^2 value for this Om
        chisq = np.sum(((model_interp - df['Hz'])/(df['dHz']))**2)
        
        # save to array and update index
        chisq_array[i,j] = chisq
        i += 1
    
    print(j)
    j+=1

end = time.perf_counter()
print(f'time to run: {end - start}')
#%%
# to save results: (make sure to change names each time it's run)

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('2D chisquared for H(z).xlsx')

# write DataFrame to excel
dataframe.to_excel(datatoexcel)

# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')




#%%

Hgrid, Omgrid = np.meshgrid(H0,Om)

chisq_array -= np.min(chisq_array)

fig = plt.figure()
ax=fig.gca()
heatmap = ax.pcolormesh(Hgrid, Omgrid, chisq_array)
contourplot = ax.contour(Hgrid, Omgrid, chisq_array, np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
fig.colorbar(heatmap)
ax.set_ylabel(r'$\Omega_{m}$', fontsize=16)
ax.set_xlabel(r'$H_0 \ km \ s^{-1} \ Mpc^{-1}$', fontsize=16)