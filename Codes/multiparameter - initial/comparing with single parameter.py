'''
We find the chisquared for the closest values in the chisquared 2D array 
to make sure ist is consistent with the 1d case
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

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

#%% for H0 = 67
#checking if we recover chisquared for the three previous H0 values we probed
# Read in Sne data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\(60-80) redone for accurate chisq.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)


columns = chisq_df.columns.tolist()[1:]

closest = float(min(columns, key=lambda x:abs(x-67000)))
ind = columns.index(closest)
chisq_67 = chisq_array[:,ind]
min_67 = np.argmin(chisq_67)
min_Om = Om[min_67]

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_67 -= np.min(chisq_67)  # define min chi^2 to be 0
in_index = np.where(chisq_67 <= Delta_squared)
chisq_67 = chisq_67[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

#plotting chisquareds
fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_67, label='$\chi^2 \ of \ model \ with \ H_0=67 \ km s^{-1} Mpc^{-1}$', color='k')

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_67)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)} for H0 = {closest}')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om, 5)}')
print(f'               - {round(min_Om - Omi[indx1][0][0], 5)}')
print('\n')



#%%
#for H0 = 70
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\(60-80) redone for accurate chisq.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)


columns = chisq_df.columns.tolist()[1:]


closest = float(min(columns, key=lambda x:abs(x-70000)))
ind = columns.index(closest)
chisq_70 = chisq_array[:,ind]
min_70 =np.argmin(chisq_70)
min_Om = Om[min_70]
# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_70 -= np.min(chisq_70)  # define min chi^2 to be 0
in_index = np.where(chisq_70 <= Delta_squared)
chisq_70 = chisq_70[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

#plotting chisquareds
fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_70, label='$\chi^2 \ of \ model \ with \ H_0=67 \ km s^{-1} Mpc^{-1}$', color='k')

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_70)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)} for H0 = {closest}')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om, 5)}')
print(f'               - {round(min_Om - Omi[indx1][0][0], 5)}')
print('\n')






#%%
#for H0 = 73
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# Read in generated array
chisq_df = pd.read_excel('data\\(60-80) redone for accurate chisq.xlsx')
chisq_array_init = np.array(chisq_df)
chisq_array = chisq_array_init[:, 1:]

# define constant
c = 3 * 10**8

# set up the model axis
H0 = np.linspace(60, 80, 300)*10**3
Om = np.linspace(0, 1, 300)
z = np.linspace(0, 1.8, 100)

#get possible valuse for H0
columns = chisq_df.columns.tolist()[1:]

#find closest value for desired H0
closest = float(min(columns, key=lambda x:abs(x-73000)))

#slice the required chisquared
ind = columns.index(closest)
chisq_73 = chisq_array[:,ind]
min_73 =np.argmin(chisq_73)
min_Om = Om[min_73]

# #############################################################################
# plot only in relevant bounds, add confidence regions
# #############################################################################

# plotting in the relevant bounds with confidence regions
Delta_squared = 20

chisq_73 -= np.min(chisq_73)  # define min chi^2 to be 0
in_index = np.where(chisq_73 <= Delta_squared)
chisq_73 = chisq_73[in_index]  # only keep in wanted region
Om = Om[in_index]  # crop Om accordingly

#plotting chisquareds
fig = plt.figure()
ax1 = fig.gca()
ax1.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax1.set_ylabel(r'$\chi^2$', fontsize=16)
ax1.set_ylim(0, 20)

ax1.plot(Om, chisq_73, label='$\chi^2 \ of \ model \ with \ H_0=67 \ km s^{-1} Mpc^{-1}$', color='k')

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om, chisq_73)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax1)

# print to user:
print('\n')
print(f'minimising \chi^2 gives a matter density = {round(min_Om, 4)} for H0 = {closest}')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om, 5)}')
print(f'               - {round(min_Om - Omi[indx1][0][0], 5)}')
print('\n')


