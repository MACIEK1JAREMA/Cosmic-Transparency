
# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = np.linspace(0,1,90)
OL = 0.77

# axis
z = np.linspace(0, 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, 1.8, 1000)  # using 1000 point for sum
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

dl1_grid = np.zeros([len(z),len(Om)])
#%%

i = 0
chisq_array = np.array([])
while i < len(Om):

    # model from list comprehension 
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1+z10[:10*j + 1])**3 + OL)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::10]) * dl1_sum
    dl1mu_model = np.array([5*np.log10(dl1_model) + 25])

    
    dl1_grid[:, i] = dl1mu_model #calculate grid of different possible model mu values
    interp = np.interp(df['z'], z, dl1_grid[:, i]) #interpolate the values in the grid as they are generated
    chisq = np.sum((interp-df['mu'])**2/(df['dmu']**2)) # calculate chisq for each dataset
    
    chisq_array = np.append(chisq_array, chisq) #create array of chisq of each Om
    
    
    i+=1
    
plt.plot(Om, chisq_array)

#min value at the following

index = chisq_array.argmin()
min_Om = Om[24]
min_chisq = chisq_array[24]


print('minimising chisquared gives a matter density of ', min_Om)
