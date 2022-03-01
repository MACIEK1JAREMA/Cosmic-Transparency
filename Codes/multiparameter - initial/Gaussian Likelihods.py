
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

#plotting Gaussian
fig = plt.figure()
ax = fig.gca()
ax.plot(H0, g)
ax.set_ylabel('$g(H_0)$', fontsize=16)
ax.set_xlabel('$H_0 \ (km s^{-1} Mpc^{-1})$', fontsize=16)
ax.set_title('$First \ Gaussian \ Prior$', fontsize=16)

# switch to likelihoods
likelihood = np.exp((-chisq_array**2)/2)
weighted = np.zeros(np.shape(likelihood))
i=0
while i < len(g):
    weighted[:,i] = likelihood[:,i]*g[i]
    i+=1
