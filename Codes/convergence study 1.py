'''
Here, we study the convergence of the approximation used in obtaining the model
from using the Riemann sum.
'''

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# Models at different accuracies superposed

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Redshift \ z$', fontsize=16)
ax.set_ylabel(r'$Luminosity \ Distance  \ d_{L} \  [Mpc]$', fontsize=16)

# Calculate the model for a rnage of accuracies in z10

# set up needed arrays
z = np.linspace(0, 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# set up an array of accuracies to use:
accuracy = np.arange(100, 2001, 100)

# loop over calculating the whole model for a few accuracies:
for accu in accuracy:
    z10 = np.linspace(0, 1.8, accu)  # using 1000 point for sum
    count10 = list(np.linspace(0, accu-1, accu).astype(int))
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z10[:int(accu/len(z))*j + 1])**3 + OL)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(accu/len(z))]) * dl1_sum
    
    # plot above models, with a legend:
    ax.plot(z, dl1_model, label=f'{accu} points in sum')
    ax.legend()

# %%

# Models sampled at 4 z values, relative errors plotted for each

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
ax3.set_xlabel(r'$Number \ of \ points $', fontsize=16)
ax4.set_xlabel(r'$Number \ of \ points $', fontsize=16)
ax1.set_ylabel(r'$relative \ error\  [\%] $', fontsize=16)
ax3.set_ylabel(r'$relative \ error\  [\%] $', fontsize=16)

# Calculate the model for a rnage of accuracies in z10, saving a few chosen dL

# set up needed arrays
z = np.linspace(0, 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# set up an array of accuracies and empty lists to store 5 smapled dL
accuracy = np.arange(100, 2001, 100)
dL_sampled = np.empty((4, len(accuracy)))
z_i = [1, 32, 67, 99]  # indexes of z at which we save dL
num = len(z_i)
i = 0  # counting variable for indexes from accu

# loop over calculating the whole model for a few accuracies:
for accu in accuracy:
    z10 = np.linspace(0, 1.8, accu)  # using 1000 point for sum
    count10 = list(np.linspace(0, accu-1, accu).astype(int))
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z10[:int(accu/len(z))*j + 1])**3 + OL)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(accu/len(z))]) * dl1_sum
    # add sampled points to saving array:
    dL_sampled[:, i] = dl1_model[z_i]
    i += 1

# from found dL_sampled, find percentage errors
change = np.empty((num, len(accuracy)-1))
for i in range(num):
    change[i, :] = abs((np.diff(dL_sampled[i, :]) / dL_sampled[i, :-1])*100)

# plot all on the made axis, with correspionding titles, semiautomatically
for i in range(0, num):
    exec('ax' + str(i+1) + '.set_title(rf\'$Probing \ at \ z= {round(z[z_i[i]], 2)}$\', fontsize=16)')
    exec('ax' + str(i+1) + '.plot(accuracy[1:], change[i, :], ' + '\'k-\' )')
    exec('ax' + str(i+1) + '.axhline(1,' + 'color=\'r\', label=r\'$1\%$\')')
    exec('ax' + str(i+1) + '.axhline(0.5,' +'color=\'g\', label=r\'$0.5\%$\' )')
    exec('ax' + str(i+1) + '.axhline(0.1,' + 'color=\'b\', label=r\'$0.1\%$\' )')
    exec('ax' + str(i+1) + '.axhline(0.01,' + 'color=\'orange\', label=r\'$0.01\%$\' )')
    exec('ax' + str(i+1) + '.legend()')
    

# %%

# Models sampled at last value of z only, plots relative error

# define constants
H0 = 70*10**3  # taking 70km s^-1 Mpc^-1
c = 3 * 10**8
Om = 0.23
OL = 0.77

# Calculate the model for a rnage of accuracies in z10, saving at chosen dL

# set up needed arrays
z = np.linspace(0, 1.8, 100)  # defining 100
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

# set up an array of accuracies and empty lists to store 5 smapled dL
accuracy = np.arange(500, 3501, 100)
dL_sampled = np.empty((len(accuracy)))
i = 0  # counting variable for indexes from accu

# loop over calculating the whole model for a few accuracies:
for accu in accuracy:
    z10 = np.linspace(0, 1.8, accu)  # using 1000 point for sum
    count10 = list(np.linspace(0, accu-1, accu).astype(int))
    dl1_sum = [(c/H0) * np.sum(1/np.sqrt(Om*(1+z10[:int(accu/len(z))*j + 1])**3 + OL)) for j in count[:]]
    dl1_model = (1+z)*z/(count10[::int(accu/len(z))]) * dl1_sum
    # save last point for this accuracy
    dL_sampled[i] = dl1_model[1]
    i += 1

# from found dL_sampled, find percentage errors
change = abs((np.diff(dL_sampled) / dL_sampled[:-1])*100)

# set up figure and visuals
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$Number \ of \ points $', fontsize=16)
ax.set_ylabel(r'$relative \ error\  [\%] $', fontsize=16)
ax.set_title(rf'$Probing \ at \ z= {round(z[1], 2)}$', fontsize=16)

# plot all on the made axis, with correspionding titles, semiautomatically
ax.plot(accuracy[1:], change[:], 'k-' )

# plot horizontal lines at 2%, 1%, 0.5% and 0.1%
ax.axhline(y=1, color='r', label=r'$1 \% $' )
ax.axhline(y=0.5, color='g', label=r'$0.5 \% $' )
ax.axhline(y=0.2, color='b', label=r'$0.2 \% $' )
ax.axhline(y=0.1, color='orange', label=r'$0.1 \% $' )
ax.legend(prop={'size': 14})
