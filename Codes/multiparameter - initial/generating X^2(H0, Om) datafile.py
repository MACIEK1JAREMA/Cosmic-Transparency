'''
Here we generate data for analysis of the likelihood function that includes
both H0 variation and Om as parameters.
We saved these into a txt file already, to avoid oversaving, we deleted that
line.
The data can be found in files 'Chisquare_array N-M' with N being lower H0
and M being the higher one, in kms^-2Mpc^-1
This code runs for about 10 mins, efficiency could be improved.
This may be attempted if time allows.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3

Om = np.linspace(0, 1, 300)

z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = np.linspace(0, len(z10)-1, len(z10)).astype(int)+1

i = 0
k = 0
chisq_array = np.zeros(np.shape(np.meshgrid(H0, Om)[0]))

while i < len(H0):
    k = 0
    while k < len(Om):
        # model from list comprehension
        dl1_sum = [(c/H0[i]) * np.sum(1/np.sqrt(Om[k]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[k] + 1)) for j in count[:]]
        dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
        # convert to mu vs z to compare to data.
        dl1mu_model = 5*np.log10(dl1_model) + 25
        
        # interpolate the values in the grid as they are generated
        interp = np.interp(df['z'], z, dl1mu_model)
        
        # get chi^2 value for this Om and save to its array
        chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
        chisq_array[k, i] = chisq
        
        k += 1
    print(i)
    i += 1

# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

#%%

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('(60-80) redone.xlsx')
  
# write DataFrame to excel
dataframe.to_excel(datatoexcel)
  
# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')

