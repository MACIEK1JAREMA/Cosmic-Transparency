'''
Here we look at including H0 as a parameter, then conpute a 2D
likelihood funciton to marginalise over H0 variation and constrain \Omega_m
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
H0 = np.linspace(70, 76, 300)*10**3

Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))
# %% DO NOT RUN THIS UNLESS YOU WANT TO BE SAT FOR 20 MINUTES!!!!!
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
    i += 1
    
# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')


# %%
#plotting chisquared stuff

# Read in data as pandas dataframe
chisq_df = pd.read_excel('data\\Chisquare_array(70-76).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

#finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_H0 = H0[index[1]]

#plotting the heatmap
fig1 = plt.figure()
ax1 = fig1.gca()
xgrid, ygrid = np.meshgrid(H0, Om)
heatmap = ax1.pcolormesh(xgrid,ygrid,chisq_array)

#adding a contour plot for clarity
contourplot = ax1.contour(xgrid,ygrid,chisq_array,np.linspace(0,1000,11), cmap = cm.jet)
ax1.clabel(contourplot)
ax1.set_ylabel('$\Omega_{m} $')
ax1.set_xlabel('$H_0 \ (m s^{-1} Mpc^{-1})$')
#plotting minimum value
ax1.plot(min_H0, min_Om, 'rx')


# now plotting a surface plot to observe the shape of chisquared
fig2 = plt.figure()
ax2 = fig2.gca(projection = '3d')
surf = ax2.plot_surface(xgrid, ygrid, chisq_array, cmap = cm.jet)
ax2.set_ylabel('$\Omega_{m} $')
ax2.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$')
ax2.set_zlabel('$\chi^2$')
#redefining tick labels so we can write H0 in standard form
ax2.set_xticklabels(["55","60","65","70","75","80"])
# %%
# switching to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

#plotting for visualisation
fig3 = plt.figure()
ax3 = fig3.gca(projection = '3d')
surf = ax3.plot_surface(xgrid, ygrid, likelihood, cmap = cm.jet)
ax3.set_ylabel('$\Omega_{m} $')
ax3.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$')
ax3.set_zlabel('likelihood')

#marginalising over H0
lik_margin = np.sum(likelihood, axis = 0)














