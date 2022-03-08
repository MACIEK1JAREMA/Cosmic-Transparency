'''
Here we proceed with first attempts of chi^2 analysis of the data and
previously devleoped model, treating only Omega_{m} (Om) as a free parameter.
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

# set up the model axis
Om = np.linspace(0, 1, 200)
z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 3000)  # inetrgal approximation axis
count10 = np.array(np.linspace(0, len(z10)-1, len(z10)).astype(int)) + 1

# set up figure and visuals for chi^2 plot
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax.set_ylabel(r'$\chi^2$', fontsize=16)

i = 0
chisq_array = np.array([])

while i < len(Om):
    # model from list comprehension
    # for each Om at const OL, curvature changes, account for it in fucntional
    # form:
    
    # for current Om and OL find the array (with z) of Ok
    Ok = np.sqrt(abs(1-Om[i]-OL)/(1+z)**2)
    
    if Om[i] + OL == 1:
        dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
        dl1_model = (1+z)*z/(np.array(count10[::int(len(z10)/len(z))])) * dl1_sum
    elif (Om[i] + OL) < 1:
        dl1_sum = [ np.sum(1/np.sqrt(Om[i]*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
        dl1_model = (1+z) *(c/H0)* Ok * np.sin(Ok*np.array(dl1_sum)*z/np.array(count10[::int(len(z10)/len(z))]))

    else:
        dl1_sum = [np.sum(1/np.sqrt(Om[i]*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
        checker = Ok*np.array(dl1_sum)*z/np.array(count10[::int(len(z10)/len(z))])
        dl1_model = (1+z) * Ok*(c/H0) * np.sinh(checker)
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    
    # save to array
    chisq_array = np.append(chisq_array, chisq)
    i += 1

# plot chi^2 against Omega_m
ax.plot(Om, chisq_array)

# get minimum value for Om and chi^2
index = chisq_array.argmin()
min_Om = Om[index]
min_chisq = chisq_array[index]

print('minimising \chi^2 gives a matter density of ', min_Om)

end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

# %%

# develop the model that corresponds to the found Om, and plot it on data
# also plot the one when using Om = 0.23

# Only run once the previous cell has been run
# to develop the model parameter first

# Plotting with data together:

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
OL = 0.77

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')

# sort it in increasing z:
df = df.sort_values('z')

# convert the mu data to d_L, make it a new column and sort w.r.t it
df_dL = 10**(0.2*df['mu'] - 5)
df.insert(4, 'dL Mpc', df_dL)
df = df.sort_values('dL Mpc')

# propagate errors and add as new column to data frame
ddL = 0.2*np.log(10)*10**(0.2*df['mu'] - 5) * df['dmu']
df.insert(5, 'ddL Mpc', ddL)

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# plot the data as errorbar plot
ax.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Calculate models:

# set up axis
z = np.linspace(0, 1.8, 100)  # for model
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, 1.8, 1000)  # for integral approximation
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))
# develop model
dl1_sum = [(c/H0) * np.sum(1/np.sqrt(min_Om*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot above models, with a legend:
ax.plot(z, dl1_model, 'g-', label=rf'$Model \ with \ \chi^2 \ minimised \ \Omega_m \ = \ {round(min_Om, 4)}$')


# plot with Om = 0.23
dl1_sum = [(c/H0) * np.sum(1/np.sqrt(0.23*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + OL)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot above models, with a legend:
ax.plot(z, dl1_model, 'r-', label=rf'$Model \ with \ \Omega_m \ = \ 0.23$')

ax.legend()

