'''
Implictly marginalising over different values for H0 using the Hubble free
distance modulus
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

# function to plot confidence intervals at given indexes
def confidence_plot(x, y, indx1, indx2, indx3, axis, colors=['r', 'g', 'b'], labels=True):
    '''
    Plots default confidence interval lines.
    The inuput y axis must be chi^2, reduced to minimum at y=0.
    
    Prameters:
        ---------------
        - x - numpy.ndarray x values
        - y - numpy.ndarray y values
        - indx1 - index of intersection with line at 1 sigma
        - indx2 - index of intersection with line at 2 sigma
        - indx3 - index of intersection with line at 3 sigma
        - ax - matplotlib axis to plot on
        - colors - colours to use for lines at the 3 intervals, in order
                   default = ['r', 'g', 'b']
        - labels - bool - default = True, defines if legend is to draw
    Returns:
        None, plots on given axis
    '''
    global ys
    
    # get values of intersections
    x1 = x[indx1]
    y1 = y[indx1]
    x2 = x[indx2]
    y2 = y[indx2]
    x3 = x[indx3]
    y3 = y[indx3]
    
    # plot horizontally
    if labels:
        axis.plot(x1, y1, color='r', ls='-.', label=r'$1 \sigma$')
        axis.plot(x2, y2, color='g', ls='-.', label=r'$2 \sigma$')
        axis.plot(x3, y3, color='b', ls='-.', label=r'$3 \sigma$')
    else:
        axis.plot(x1, y1, color='r', ls='-.')
        axis.plot(x2, y2, color='g', ls='-.')
        axis.plot(x3, y3, color='b', ls='-.')
    
    # organise intersections into a list for loop
    xs = [x1[0], x1[1], x2[0], x2[1], x3[0], x3[1]]
    ys = [y1[0], y1[1], y2[0], y2[1], y3[0], y3[1]]
    
    # loop over plotting each vertical line
    for i in range(3):
        axis.axvline(xs[2*i], ymin=0, ymax=(ys[2*i]-axis.get_ybound()[0])/axis.get_ybound()[1], color=colors[i], ls='-.')
        axis.axvline(xs[2*i+1], ymin=0, ymax=(ys[2*i+1]-axis.get_ybound()[0])/axis.get_ybound()[1], color=colors[i], ls='-.')
        
    if labels:
        axis.legend()
    else:
        pass


# %%

# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 73*10**3  # unimportant value here, marginalised over anyway
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(0.001, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0.001, 1.8, 1000)  # inetrgal approximation axis
count10 = np.array((np.linspace(0, len(z10)-1, len(z10)).astype(int))) +1

# develop models for each Om, get it's theoretical M and chi^2
i = 0
chisq_array = np.array([])
models_mu = np.zeros((len(df['z']), len(Om)))

while i < len(Om):
    # model from list comprehension
    dl_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
    dl_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl_sum
    
    # interpolate the values to match data size
    dl_model_interp = np.interp(x=df['z'], xp=z, fp=dl_model)
    
    # define theoretical absolute magnitude from these and use it for model in mu
    M = np.sum((df['mu'] - 5*np.log10(dl_model_interp)) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
    mu_model_interp = 5*np.log10(dl_model_interp) + M
    print(M)
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((mu_model_interp - df['mu'])**2/(df['dmu'])**2))
    chisq_array = np.append(chisq_array, chisq)
    
    models_mu[:, i] = mu_model_interp

    i += 1


# plot chi^2 initially
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_ylabel(r'$\chi^2$', fontsize=16)
ax.plot(Om, chisq_array)

# plot model in mu with minium chi^2
index_min = np.where(chisq_array == np.min(chisq_array))[0][0]
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$z$', fontsize=16)
ax1.set_ylabel(r'$\mu$', fontsize=16)
for i in [100, 150, 200, 250, 299]:
    ax1.plot(df['z'], models_mu[:, i], label=fr'$Model \ \Omega_m ={round(Om[i], 4)}$')  # interpolated z
ax1.plot(df['z'], models_mu[:, index_min], label=rf'$minimum \ \chi^2 \ model \ \Omega_m ={round(Om[index_min], 4)}$')

ax1.legend()

# define relevant chi^2 regions and plot there only with confidence:
Delta_squared = 20
chisq_array = chisq_array - np.min(chisq_array)
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig2 = plt.figure()
ax2 = fig2.gca()
ax2.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax2.set_ylabel(r'$\chi^2$', fontsize=16)
ax2.set_ylim(0, 20)
ax2.plot(Om, chisq_array)

# add plot of convergence:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_array)
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax2)

print('\n')
print('minimum value was found to be at Omega_m = ')
print(Om[np.where(chisq_array == np.min(chisq_array))[0][0]])
print('\n')
print('the starndard deviation was found to be:')
print(f' + {Omi[indx1[1]] - Om[np.where(chisq_array == np.min(chisq_array))[0][0]]}')
print(f' - {Om[np.where(chisq_array == np.min(chisq_array))[0][0]] - Omi[indx1[0]]}')

#%%
#Plotting default chisquared for different M to illustrate dependence (can delete later)



# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))

i = 0
k = 0
chisq_array = np.array([])
M = np.array([24.5,24.9,25.1,25.5])


fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
test = M[3]
while k < len(M):
    while i < len(Om):
        # model from list comprehension
        dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
        dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum
        # convert to mu vs z to compare to data.
        dl1mu_model = 5*np.log10(dl1_model) + M[k]
        
        # interpolate the values in the grid as they are generated
        interp = np.interp(df['z'], z, dl1mu_model)
        
        # get chi^2 value for this Om and save to its array
        chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
        chisq_array = np.append(chisq_array, chisq)
        
        i += 1
    
    plt.plot(Om, chisq_array, label=fr'$M = {round(M[k], 4)}$')
    print(M[k])
    k = k + 1
    i=0
    chisq_array = np.array([])

plt.title('$\chi^2 \ of \ model \ with \ H_0=70 \ km s^{-1} Mpc^{-1}$')
ax1.legend()
































