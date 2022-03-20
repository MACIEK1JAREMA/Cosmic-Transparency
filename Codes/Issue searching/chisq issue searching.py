'''

Tests of chi^2 calculations

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import Codes.Module as module

# %%

# #############################################################################
# Method 1 - our integration
# #############################################################################

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = 0.20
z = np.linspace(0, 0.3, 400)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z1000 = np.linspace(0, z, 1000)  # inetrgal approximation axis



# model from list comprehension
combs = [1/np.sqrt(Om*(1+z1000[:,j])**3 - Om + 1) for j in count[:]]
dl_sum = np.sum(combs, axis=1)
dl_model = (c/H0)*(1+z)*z/1000 * dl_sum


# plot the model
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$z$', fontsize=20)
ax.set_ylabel(r'$d_L \ [Mpc]$', fontsize=20)
ax.plot(z, dl_model, label=r'$Riemann \ summation \ Model \ with \ \Omega_{m}=0.2$')

# #############################################################################
# Method 2 - approximate integral in our range of parameters
# #############################################################################

dl_model = 2*c/(3*H0*Om) * (1+z) * (np.sqrt(1 + 3*Om*z) - 1)


# plot the model
ax.set_xlabel(r'$z$', fontsize=20)
ax.set_ylabel(r'$d_L \ [Mpc]$', fontsize=20)
ax.plot(z, dl_model, label=r'$Approximate \ inetrgal \ model \ with \ \Omega_{m}=0.2$')

ax.legend(fontsize=18)

