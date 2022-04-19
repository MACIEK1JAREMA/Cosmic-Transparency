
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
H0 = np.linspace(70, 76, 300)*10**3

Om = np.linspace(0, 1, 300)

z = np.linspace(np.min(df['z']), 1.8, 500)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, z, 500)  # inetrgal approximation axis

i = 0
k = 0
chisq_array = np.zeros(np.shape(np.meshgrid(H0, Om)[0]))

while i < len(H0):
    k = 0
    while k < len(Om):
        # model for d_{L}
        combs = 1/np.sqrt(Om[k]*(1+z10)**3 - Om[k] + 1)
        dl1_sum = np.sum(combs, axis=0)
        dl1_model = (c/H0[i])*(1+z)*z/500 * dl1_sum
        
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

# %%

# to save results: (make sure to change names each time it's run)

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('(70-76) chisq(Om, H0).xlsx')

# write DataFrame to excel
dataframe.to_excel(datatoexcel)

# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')
