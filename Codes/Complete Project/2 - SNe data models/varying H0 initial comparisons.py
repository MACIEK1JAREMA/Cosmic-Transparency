'''
Here we look at the effect that a changed value of H0 has on the end models
we do so on the example of k=0 models with chi^2 analysis
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Define constants
H0_lambdacdm = 67*10**3  # H0 predicted by LCDM and calibrated by Plank
H0_SHOES = 73*10**3  # Distance ladder measurement by SH0ES
H0_mean = 70*10**3  # mean
c = 3 * 10**8  # light speed

# define relevant chi^2 regions
Delta_squared = 20

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

# Set up the plot for the model and data + visuals
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax1.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)
# plot data
ax1.errorbar(df['z'], df['dL Mpc'], yerr=df['ddL Mpc'],
            capsize=2, fmt='.', markersize=5, ecolor='k')

# Set up a plot for chi^2 analysis + visuals
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.set_xlabel(r'$\Omega_{m} $', fontsize=16)
ax2.set_ylabel(r'$\chi^2$', fontsize=16)
ax2.set_ylim(0, 20)

# set up the model axis
Om = np.linspace(0, 1, 300)
z = np.linspace(np.min(df['z']), 1.8, 100)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)
z1000 = np.linspace(0, z, 1000)  # inetrgal approximation axis


# ############################################################################
# complete the model for LCDM H0
# ############################################################################

i = 0
chisq_array_LCDM = np.array([])

while i < len(Om):
    # model from list comprehension
    combs = [1/np.sqrt(Om[i]*(1+z1000[:,j])**3 - Om[i] + 1) for j in count[:]]
    dl1_sum = np.sum(combs, axis = 1)
    dl1_model = (c/H0_lambdacdm)*(1+z)*z/1000 * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_LCDM = np.append(chisq_array_LCDM, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_LCDM.argmin()
min_Om_LCDM = Om[index]
min_chisq_LCDM = chisq_array_LCDM[index]

# reduce it the chi^2
minimum_chi2_LCDM = np.min(chisq_array_LCDM)
chisq_array_LCDM -= minimum_chi2_LCDM  # define min chi^2 to be 0
in_index = np.where(chisq_array_LCDM <= Delta_squared)
chisq_array_LCDM = chisq_array_LCDM[in_index]  # only keep in wanted region
Om_LCDM = Om[in_index]  # crop Om accordingly

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om_LCDM, chisq_array_LCDM)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax2)

# develop the model on this found Om

combs = [1/np.sqrt(min_Om_LCDM*(1+z1000[:,j])**3 - min_Om_LCDM + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0_lambdacdm)*(1+z)*z/1000 * dl1_sum

# plot it on the data and chi^2 against Omega_m for it
ax1.plot(z, dl1_model, 'b-', label=rf'$H_0 \ = \ 67 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_LCDM, 4)}$')
ax2.plot(Om_LCDM, chisq_array_LCDM, 'b-', label = '$H_0 \ = \ 67 \ km \ s^{-1} \ Mpc^{-1}$')

print(f'minimising \chi^2 for H0=67 kms-1Mpc-1 gives a matter density of {round(min_Om_LCDM, 4)}')
print('\n')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om_LCDM, 5)}')
print(f'               - {round(min_Om_LCDM - Omi[indx1][0][0], 5)}')
print('\n')

# ############################################################################
# Repeat for other H0
# ############################################################################

i = 0
chisq_array_SHOES = np.array([])

while i < len(Om):
    # model from list comprehension
    combs = [1/np.sqrt(Om[i]*(1+z1000[:,j])**3 - Om[i] + 1) for j in count[:]]
    dl1_sum = np.sum(combs, axis = 1)
    dl1_model = (c/H0_SHOES)*(1+z)*z/1000 * dl1_sum
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_SHOES = np.append(chisq_array_SHOES, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_SHOES.argmin()
min_Om_SHOES = Om[index]
min_chisq_SHOES = chisq_array_SHOES[index]

# reduce it the chi^2
minimum_chi2_SHOES = np.min(chisq_array_SHOES)
chisq_array_SHOES -= minimum_chi2_SHOES  # define min chi^2 to be 0
in_index = np.where(chisq_array_SHOES <= Delta_squared)
chisq_array_SHOES = chisq_array_SHOES[in_index]  # only keep in wanted region
Om_SHOES = Om[in_index]  # crop Om accordingly

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om_SHOES, chisq_array_SHOES)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax2, labels=False)

# develop the model on this found Om

combs = [1/np.sqrt(min_Om_SHOES*(1+z1000[:,j])**3 - min_Om_SHOES + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0_SHOES)*(1+z)*z/1000 * dl1_sum
# plot chi^2 against Omega_m
ax1.plot(z, dl1_model, 'r-', label=rf'$H_0 \ = \ 73 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_SHOES, 4)}$')
ax2.plot(Om_SHOES, chisq_array_SHOES, 'r-', label = '$H_0 \ = \ 73 \ km \ s^{-1} \ Mpc^{-1}$')

print(f'minimising \chi^2 for H0=73 kms-1Mpc-1 gives a matter density of {round(min_Om_SHOES, 4)}')
print('\n')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om_SHOES, 5)}')
print(f'               - {round(min_Om_SHOES - Omi[indx1][0][0], 5)}')
print('\n')

# ############################################################################
# repeat for the mean, as used before, to compare.
# ############################################################################

i = 0
chisq_array_mean = np.array([])

while i < len(Om):
    # model from list comprehension
    combs = [1/np.sqrt(Om[i]*(1+z1000[:,j])**3 - Om[i] + 1) for j in count[:]]
    dl1_sum = np.sum(combs, axis = 1)
    dl1_model = (c/H0_mean)*(1+z)*z/1000 * dl1_sum
    # convert to mu vs z to compare to data.
    dl1mu_model = 5*np.log10(dl1_model) + 25
    
    # interpolate the values in the grid as they are generated
    interp = np.interp(df['z'], z, dl1mu_model)
    
    # get chi^2 value for this Om and save to its array
    chisq = np.sum(((interp - df['mu'])/(df['dmu']))**2)
    chisq_array_mean = np.append(chisq_array_mean, chisq)
    
    i += 1

# get minimum value for Om and chi^2
index = chisq_array_mean.argmin()
min_Om_mean = Om[index]
min_chisq_mean = chisq_array_mean[index]

# reduce it the chi^2
minimum_chi2_mean = np.min(chisq_array_mean)
chisq_array_mean -= minimum_chi2_mean  # define min chi^2 to be 0
in_index = np.where(chisq_array_mean <= Delta_squared)
chisq_array_mean = chisq_array_mean[in_index]  # only keep in wanted region
Om_mean = Om[in_index]  # crop Om accordingly

# interpolate:
Omi = np.linspace(0, 1, 10000)
chi_sqr_i = np.interp(np.linspace(0, 1, 10000), Om_mean, chisq_array_mean)

# get intercept indexes
indx1 = np.argwhere(np.diff(np.sign(chi_sqr_i - np.ones(np.shape(chi_sqr_i)))))
indx2 = np.argwhere(np.diff(np.sign(chi_sqr_i - 2.71*np.ones(np.shape(chi_sqr_i)))))
indx3 = np.argwhere(np.diff(np.sign(chi_sqr_i - 9*np.ones(np.shape(chi_sqr_i)))))

# plot confidence regions
confidence_plot(Omi, chi_sqr_i, indx1, indx2, indx3, ax2, labels=False)


# develop the model on this found Om
combs = [1/np.sqrt(min_Om_mean*(1+z1000[:,j])**3 - min_Om_mean + 1) for j in count[:]]
dl1_sum = np.sum(combs, axis = 1)
dl1_model = (c/H0_mean)*(1+z)*z/1000 * dl1_sum

# plot chi^2 against Omega_m
ax1.plot(z, dl1_model, 'k-', label=rf'$H_0 \ = \ 70 \ km \ s^{-1} \ Mpc^{-1} \ Model \ with \ \Omega_m \ = \ {round(min_Om_mean, 4)}$')
ax2.plot(Om_mean, chisq_array_mean, 'k-', label = '$H_0 \ = \ 70 \ km \ s^{-1} \ Mpc^{-1}$')

print(f'minimising \chi^2 for H0=70 kms-1Mpc-1 gives a matter density of {round(min_Om_mean, 4)}')
print('\n')
print('1-sigma error =')
print(f'               + {round(Omi[indx1][1][0] - min_Om_mean, 5)}')
print(f'               - {round(min_Om_mean - Omi[indx1][0][0], 5)}')
print('\n')

# finilise plot legends
ax2.legend(loc='upper left')
ax1.legend()
