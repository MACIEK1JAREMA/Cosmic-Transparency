'''
Here we proceed with first attempts of chi^2 analysis of the data and
previously devleoped model, treating only Omega_{m} (Om) as a free parameter.
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# function to plot confidence intervals at given indexes
def confidence_plot(x, y, indx1, indx2, indx3, ax, colors=['r', 'g', 'b']):
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
    Returns:
        None, plots on given axis
    '''
    
    # get values of intersections
    x1 = x[indx1]
    y1 = y[indx1]
    x2 = x[indx2]
    y2 = y[indx2]
    x3 = x[indx3]
    y3 = y[indx3]
    
    # plot horizontally
    ax.plot(x1, y1, color='r', ls='-.', label=r'$1 \sigma$')
    ax.plot(x2, y2, color='g', ls='-.', label=r'$2 \sigma$')
    ax.plot(x3, y3, color='b', ls='-.', label=r'$3 \sigma$')
    
    # organise intersections into a list for loop
    xs = [x1[0], x1[1], x2[0], x2[1], x3[0], x3[1]]
    ys = [y1[0], y1[1], y2[0], y2[1], y3[0], y3[1]]
    
    # loop over plotting each line
    for i in range(3):
        ax.axvline(xs[2*i], ymin=0, ymax=(ys[2*i]-ax.get_ybound()[0])/ax.get_ybound()[1], color=colors[i], ls='-.')
        ax.axvline(xs[2*i+1], ymin=0, ymax=(ys[2*i+1]-ax.get_ybound()[0])/ax.get_ybound()[1], color=colors[i], ls='-.')

    ax.legend()

# %%

start_t = time.perf_counter()

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(0.001, 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, 1.8, 1000)  # inetrgal approximation axis
count10 = np.array(list(np.linspace(0, len(z10)-1, len(z10)).astype(int)))+1

i = 0
chisq_array = np.array([])

while i < len(Om):
    # model from list comprehension
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om[i]*(1 + z10[:int(len(z10)/len(z))*j + 1])**3 - Om[i] + 1)) for j in count[:]]
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
min_chisq = chisq_array[index]

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
in_index = np.where(chisq_array <= Delta_squared)
chisq_array = chisq_array[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)

ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_array, label='$\chi^2 \ of \ model \ with \ H_0=70 \ km s^{-1} Mpc^{-1}$', color='k')

# #############################################################################
# find corresponding Om at confidence region boundaries
# #############################################################################

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_array)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)}')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om, 5)}')
print(f'               - {round(min_Om - Omi[indx1][0][0], 5)}')
print('\n')

# time to run
end_t = time.perf_counter()
print(f'time to run: {round(end_t - start_t, 5)} s')

# %%

# develop the model that corresponds to the found Om, and plot it o data
# also plot the one when using Om = 0.23

# Plotting with data together:

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8

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
ax1 = fig.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# plot the data as errorbar plot
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Calculate models:

# set up axis
z = np.linspace(0, 1.8, 100)  # for model
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z10 = np.linspace(0, 1.8, 1000)  # for integral approximation
count10 = list(np.linspace(0, len(z10)-1, len(z10)).astype(int))
# develop model
dl1_sum = [(c/H0) * np.sum(1/np.sqrt(min_Om*(1+z10[:int(len(z10)/len(z))*j + 1])**3 - min_Om + 1)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot above models, with a legend:
ax1.plot(z, dl1_model, 'g-', label=rf'$Model \ with \ \chi^2 \ minimised \ \Omega_m \ = \ {round(min_Om, 4)}$')


# plot with Om = 0.23
dl1_sum = [(c/H0) * np.sum(1/np.sqrt(0.23*(1+z10[:int(len(z10)/len(z))*j + 1])**3 + 0.77)) for j in count[:]]
dl1_model = (1+z)*z/(count10[::int(len(z10)/len(z))]) * dl1_sum

# plot above models, with a legend:
ax1.plot(z, dl1_model, 'r-', label=rf'$Model \ with \ \Omega_m \ = \ 0.23$')

ax1.legend()
