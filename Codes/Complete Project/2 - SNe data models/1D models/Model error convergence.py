
# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# Models at different accuracies superposed

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # sort it in increasing z

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$Redshift \ z$', fontsize=20)
ax.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=20)

# Calculate the model for a rnage of accuracies in z10

# set up needed arrays
z = np.linspace(np.min(df['z']), 1.8, 100) # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# set up an array of accuracies to use:
accuracy = np.arange(50, 500, 50)

# loop over calculating the whole model for a few accuracies:
for accu in accuracy:
    z10 = np.linspace(0, z, accu)  # using accu points for sum
    
    # model for d_L
    int_arg = 1/np.sqrt(Om*(1+z10)**3 + 1 - Om)  # integrand
    dl_sum = np.sum(int_arg, axis=0)  # integral result
    dl_model = (c/H0)*(1+z)*(z/accu) * dl_sum  # model

    # plot above models, with a legend:
    ax.plot(z, dl_model, label=f'{accu} points in sum')
    
    ax.legend(fontsize=16)

# %%

# Models sampled at 4 z values, relative errors plotted for each

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # sort it in increasing z

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# set up figure and visuals
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.tick_params(labelsize=16)
ax2.tick_params(labelsize=16)
ax3.tick_params(labelsize=16)
ax4.tick_params(labelsize=16)
ax3.set_xlabel(r'$Number \ of \ points $', fontsize=20)
ax4.set_xlabel(r'$Number \ of \ points $', fontsize=20)
ax1.set_ylabel(r'$relative \ error\  [\%] $', fontsize=20)
ax3.set_ylabel(r'$relative \ error\  [\%] $', fontsize=20)

# Calculate the model for a rnage of accuracies in z10, saving a few chosen dL

# set up needed arrays
z = np.linspace(np.min(df['z']), 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# set up an array of accuracies and empty lists to store 5 smapled dL
accuracy = np.arange(2, 40, 2)
dL_sampled = np.empty((4, len(accuracy)))
z_i = [2, 32, 67, 77]  # indexes of z at which we save dL
num = len(z_i)
i = 0  # counting variable for indexes from accu

# loop over calculating the whole model for a few accuracies:
for accu in accuracy:
    z10 = np.linspace(0, z, accu)  # z' array for intergal approx.
    
    # model for d_L
    int_arg = 1/np.sqrt(Om*(1+z10)**3 + 1 - Om)  # integrand
    dl_sum = np.sum(int_arg, axis=0)  # integral result
    dl_model = (c/H0)*(1+z)*(z/accu) * dl_sum  # model
    
    # add sampled points to saving array:
    dL_sampled[:, i] = dl_model[z_i]
    i += 1

# from found dL_sampled, find percentage errors
change = np.empty((num, len(accuracy)-1))
for i in range(num):
    change[i, :] = abs((np.diff(dL_sampled[i, :]) / dL_sampled[i, :-1])*100)


# #############################################################################
# for each z_i, find average error in the data points at z nearby
# #############################################################################

# convert the mu data to d_L, make it a new column and sort w.r.t it
df_dL = 10**(0.2*df['mu'] - 5)
df.insert(4, 'dL Mpc', df_dL)
df = df.sort_values('dL Mpc')

# propagate errors and add as new column to data frame
ddL = 0.2*np.log(10)*10**(0.2*df['mu'] - 5) * df['dmu']
df.insert(5, 'ddL Mpc', ddL)

# set up interval within which to classify data, as: 'at nearby z'
# use half the spacing of our array for this
dz = 0.5*(z[1] - z[0])

# at each z find nearby points, their relative errors and average, then save
data_errs = np.array([])  # to save
for zi in z[z_i][:3]:
    indexes_in = np.where((df['z'] > zi-dz) & (df['z'] < zi + dz))[0]
    data_in = df['mu'][indexes_in]
    errors_in = df['dmu'][indexes_in]
    rels_in = errors_in/data_in
    data_err = np.mean(rels_in)
    data_errs = np.append(data_errs, data_err*100)  # in %

# last index:
data_in = df['mu'][-7:]
errors_in = df['dmu'][-7:]
rels_in = errors_in/data_in
data_err = np.mean(rels_in)
data_errs = np.append(data_errs, data_err*100)  # in %

# #############################################################################
# plot results
# #############################################################################

# plot all on the made axis, with correspionding titles, semiautomatically
for i in range(0, num):
    exec('ax' + str(i+1) + '.set_title(rf\'$Probing \ at \ z= {round(z[z_i[i]], 2)}$\', fontsize=20)')
    exec('ax' + str(i+1) + '.plot(accuracy[1:], change[i, :], ' + '\'k-\' )')
    exec('ax' + str(i+1) + '.axhline(data_errs[i],' + ' color=\'c\', ls=\'-.\', label=rf\'$ data \ error \ = \ {round(data_errs[i], 2)} \% $\' )')
    exec('ax' + str(i+1) + '.axhline(1,' + ' color=\'r\', ls=\'-.\', label=r\'$1\%$\')')
    exec('ax' + str(i+1) + '.axhline(0.5,' +' color=\'g\', ls=\'-.\', label=r\'$0.5\%$\' )')
    exec('ax' + str(i+1) + '.axhline(0.1,' + ' color=\'b\', ls=\'-.\', label=r\'$0.1\%$\' )')
    exec('ax' + str(i+1) + '.axhline(0.01,' + ' color=\'orange\', ls=\'-.\', label=r\'$0.01\%$\' )')
    exec('ax' + str(i+1) + '.legend(fontsize=16)')
