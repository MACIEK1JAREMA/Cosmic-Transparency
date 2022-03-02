
'''
We import the generated chi^2 data from models that include varying H0 and Om
We plot their chi^2 and likelihood function.
Then attempt to marginalise over H0 using a gaussian prior
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

def integrate(F, x_min, x_max):
    value_strip = 0
    value_total = 0
    width = (x_max - x_min)/np.size(F[0,:])
    i = 0
    for i in range(np.size(F[0,:])):
        value_strip = width*F[i]
        value_total += value_strip
    return value_total


# %%

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\Chisquare_array(70-76).xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(70, 76, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)

# finding minimum of chisquared coords
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0

#defining Gaussian
g = np.exp(-(H0/1000-73)**2/2)/np.sqrt(2*np.pi)


# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)

#multipying likelihood by Gaussian
weighted = np.zeros(np.shape(likelihood))
i=0
while i < len(g):
    weighted[:,i] = likelihood[:,i]*g[i]
    i+=1

#marginalising over H0
lik_margin = np.trapz(weighted, x = H0, axis = 1) #integrating along columns(along H0)

#trying maciej's integration formula (NEED TO FINISH THIS!!!!)
#test = integrate(weighted, H0[0], H0[-1])

#normalising
lik_margin /= np.sum(lik_margin)

Om_found = Om[np.where(lik_margin == np.max(lik_margin))[0]]

# find peak value and where 68.3% of it lies for 1 \sigma error
variables = st.rv_discrete(values=(Om, lik_margin))
confidence1 = variables.interval(0.683)[1] - Om_found
confidence2 = Om_found - variables.interval(0.683)[0]

print(f'Om = {round(Om_found[0], 5)}')
print('\n')
print(f'with confidence: \n')
print(f'                +{round(confidence1[0], 6)}')
print(f'                -{round(confidence2[0], 6)}')

fig = plt.figure()
ax = fig.gca()
ax.plot(Om[60:90], lik_margin[60:90])
ax.set_ylabel(r'$Likelihood \ marginalised \ over \ H_{0} \ L(\Omega_{m})$', fontsize=16)
ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_title('$Marginalised \ Gaussian \ Likelihood \ Function$', fontsize=16)

#%% plotting stuff

#plotting Gaussian
fig = plt.figure()
ax = fig.gca()
ax.plot(H0, g)
ax.set_ylabel('$g(H_0)$', fontsize=16)
ax.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$', fontsize=16)
ax.set_title('$First \ Gaussian \ Prior$', fontsize=16)

# plot in 3D, but restrict range:
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_ylim(0.2, 0.3)
Hgrid, Omgrid = np.meshgrid(H0, Om[60:90])
weighted = weighted[60:90, :]
surf = ax2.plot_surface(Hgrid/1000, Omgrid, weighted, cmap=cm.jet)
ax2.set_ylabel('$\Omega_{m} $', fontsize=16)
ax2.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$', fontsize=16)
ax2.set_zlabel('likelihood', fontsize=16)






