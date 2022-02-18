'''
We quickly side step to look at a convergence study in z
when finding its minimum with chi^2 analysis.
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
OL = 0.77
Om = np.linspace(0, 1, 400)

# set up the model axis
z10 = np.linspace(0, 1.8, 3000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

accuracy = [100, 120, 125, 150, 200, 250, 300, 375, 500, 600]

Om_result = []

for accu in accuracy:
    i = 0
    chisq_array = np.array([])
    
    z = np.linspace(0, 1.8, accu)
    count = np.linspace(0, len(z)-1, len(z)).astype(int)
    count = list(count)
    
    while i < len(Om):
        # model from list comprehension
        dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
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
    
    # add to overall analysis array
    Om_result.append(min_Om)

# get relative error and plot
error = abs((np.diff(Om_result) / Om_result[:-1])*100)

fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Number \ of \ points \ in \ z $', fontsize=16)
ax.set_ylabel(r'$relative \ error \ in \ found \ \Omega_{m} \ [\%]$', fontsize=16)
plt.plot(accuracy[1:], error, 'k-')

ax.axhline(1, color='r', label=r'$1 \%$')
ax.axhline(0.5, color='g', label=r'$0.5 \%$')
ax.axhline(0.2, color='b', label=r'$0.2 \%$')
ax.axhline(0.1, color='orange', label=r'$0.1 \%$')

ax.legend()

end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')
