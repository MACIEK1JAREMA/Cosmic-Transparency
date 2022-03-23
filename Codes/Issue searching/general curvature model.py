'''
We look at generalising our moodels for any curvature, implicitly marginalising
over H_0. Looking to reproduce N. Suzuki 2011 results
'''

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Codes.Module as module
import matplotlib.cm as cm

# %%

# Read in data as pandas dataframe
df = pd.read_excel('data\\SNe data.xlsx')
df = df.sort_values('z')  # increasing z sort

# define constants
H0 = 70*10**3
c = 3 * 10**8

# set up the model axis
OL = np.linspace(0.2, 1.1, 500)
Om = np.linspace(0, 0.6, 500)
z = np.linspace(np.min(df['z']), 1.8, 300)
count = np.linspace(0, len(z)-1, len(z)).astype(int)
count = list(count)

z10 = np.linspace(0, z, 1000)  # inetrgal approximation axis

i = 0
chisq_array = np.zeros(np.shape(np.meshgrid(Om, OL)[0]))
# rows are Om, columns are OL

while i < len(Om):
    j = 0
    while j < len(OL):
        # model from list comprehension
        # for each Om at const OL, curvature changes, account for it in fucntional
        # form
        
        # for current Om and OL find the array (with z) of Ok
        sqOk = np.sqrt(abs(1-Om[i]-OL[j]))
        
        # get the integration evaulated, this is part of the arg to sin/sinh
        # so same for each case
        int_arg = [1/np.sqrt(Om[i]*(1+z10[:, k])**3 + sqOk**2*(1+z10[:, k])**2 + OL[j]) for k in count[:]]
        dl_sum = np.sum(int_arg, axis=1)
        
        # develop dL from integrated expression, depeding on curvature.
        if Om[i] + OL[j] == 1:
            dl_model = (c/H0) * (1+z)*z/1000 * dl_sum
        elif (Om[i] + OL[j]) < 1:
            dl_model = (c/H0) * (1+z) / sqOk * np.sin(sqOk*dl_sum*z/1000)
        else:
            dl_model = (1+z) / sqOk * (c/H0) * np.sinh(sqOk*dl_sum*z/1000)
        
        # interpolate the values to match data size
        dl_model_interp = np.interp(x=df['z'], xp=z, fp=dl_model)
        
        # define theoretical absolute magnitude from these and use it for model in mu
        M = np.sum((df['mu'] - 5*np.log10(dl_model_interp)) / (df['dmu']**2)) / np.sum(1/(df['dmu']**2))
        
        mu_model_interp = 5*np.log10(dl_model_interp) + M
        
        # get chi^2 value for this Om
        chisq = np.sum(((mu_model_interp - df['mu'])/(df['dmu']))**2)
        
        # save to array
        chisq_array[i, j] = chisq
        
        j += 1
    
    print(f'Completed i={i} out of {len(Om)}')
    i += 1

# %%

# plotting found chi^2

# set up figure and visuals for chi^2 plot
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_ylabel(r'$\Omega_{\Lambda}$', fontsize=16)

# plot chi^2 as heatmap and then add contours
chisq_array -= np.min(chisq_array)
Omgrid, OLgrid = np.meshgrid(Om, OL)
heatmap = ax.pcolormesh(Omgrid, OLgrid, chisq_array.transpose())
contourplot = ax.contour(Omgrid, OLgrid, chisq_array.transpose(), np.array([2.30, 4.61, 11.8]), cmap=cm.jet)
#ax1.clabel(contourplot)
fig.colorbar(heatmap)


# get minimum value for Om and chi^2
print(np.min(chisq_array))
chisq_array -= np.min(chisq_array)  # define min chi^2 to be 0
index = np.unravel_index(np.argmin(chisq_array, axis=None), chisq_array.shape)
min_Om = Om[index[0]]
min_OL = OL[index[1]]
print(f'Minimum Om is: {min_Om}')
print('\n')
print(f'Minimum OL is: {min_OL}')


# %%

# saving results
'''(make sure to change names each time it's run)'''

dataframe = pd.DataFrame(chisq_array)

# writing to Excel
datatoexcel = pd.ExcelWriter('gen curvature chi^2 (300, 300).xlsx')

# write DataFrame to excel
dataframe.to_excel(datatoexcel)

# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')
