'''
Here, we develop and test a function that will find confidence regions
of N dimensionsal (1, 2, 3) functions and optionally plot them
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sciint
import matplotlib.cm as cm


# funciton for 2D integration
def integrate2D(F, xg, yg, indx_1x=0, indx_2x=-1, indx_1y=0, indx_2y=-1, interp=0, interp_kind='linear'):
    '''
    Integrates over a 2D fucntion F
    
    Parameters:
        ----------
    -- F - 2D np.ndarray to integrate
    -- xg - 2D np.ndarray grid of x values F is evaluated at
    -- yg - 2D np.ndarray grid of y values F is evaluated at
    -- indx_1x - lower limit in x as index to that corresponding value in xg,
                 default = 0, first value
    -- indx_2x - upper limit in x as index to that corresponding value in xg,
                 default = -1, last value
    -- indx_1y - upper limit in y as index to that corresponding value in yg,
                 default = 0, first value
    -- indx_2y - upper limit in y as index to that corresponding value in yg,
                 default = -1, last value
    -- interp - int, if > 10, we interpolate grids and F to that size,
                 default = 0
    -- interp_kind - str - scipy interpolation kind parameter to pass in,
                 default  = 'linear'
    
    Returns:
        ----------
    --- result value - float
    
    '''
    
    # deal with interpolating if needed
    if interp > 10:
        F_interp = sciint.interp2d(xg[0, :], yg[:, 0], F, interp_kind)
        x = np.linspace(xg[0, 0], xg[0, -1], interp)
        y = np.linspace(yg[0, 0], yg[-1, 0], interp)
        xg, yg = np.meshgrid(x, y)
        F = F_interp(x, y)
        
    # get numerical dx and dy
    dx = (xg[0, indx_2x] - xg[0, indx_1x])/np.size(F[:, 0])
    dy = (yg[indx_2y, 0] - yg[indx_1y, 0])/np.size(F[0, :])
    
    # sum over values in F with these dx and dy
    result = np.sum(F*dx*dy)
    
    return result


def confidence(F, xg, yg, accu, interp=0, interp_kind='linear'):
    '''
    Find heights corresponding to confidence levels of a 2D function:
    
    Prameters:
        ----------
    -- F - 2D np.ndarray to integrate
    -- xg - 2D np.ndarray grid of x values F is evaluated at
    -- yg - 2D np.ndarray grid of y values F is evaluated at
    -- accuracy to use in height, to small will be inacurate, too large will
        raise errors, as no points will lie in F above values 1 step
        lower than maximum
    -- interp - int, if > 10, we interpolate grids and F to that size,
                 default = 0
    -- interp_kind - str - scipy interpolation kind parameter to pass in,
                 default  = 'linear'
    
    Returns:
        ----------
    --- list of heights in order [3sigma, 2sigma, 1sigma]  - in this order
        for contours to be in ascending order for matplotlib.
    
    '''
    
    global integrals, heights
    
    # interpolate F if user wanted to:
    if interp > 10:
        F_interp = sciint.interp2d(xg[0, :], yg[:, 0], F, interp_kind)
        x = np.linspace(xg[0, 0], xg[0, -1], interp)
        y = np.linspace(yg[0, 0], yg[-1, 0], interp)
        xg, yg = np.meshgrid(x, y)
        F = F_interp(x, y)
    
    # find the maximum index of F and get corr height.
    max_indx = np.where(F == np.max(F))
    F_max = F[max_indx[0][0], max_indx[1][0]]
    
    # set up an array of heights to sift through:
    heights = np.linspace(F_max, np.min(F), accu)
    
    # find 2D integrals from max height to selected height:
    integrals = []
    for h in heights[1:]:
        F_integrate = (F > h) * F  # make everything smaller than h=0, no integral contribution
        value = integrate2D(F_integrate, xg, yg)  # by above line, can integrate whole domain
        integrals.append(value)
    
    # find where it's nearest to each sigma thershold and return
    
    # 1-sigma:
    integrals = np.array(integrals)
    s1_indx = np.where(np.diff(np.sign(integrals - 0.6826)))[0]
    s1_height = heights[s1_indx][0]
    
    # 2-sigma:
    integrals = np.array(integrals)
    s2_indx = np.where(np.diff(np.sign(integrals - 0.9546)))[0]
    s2_height = heights[s2_indx][0]
    
    # 3-sigma:
    integrals = np.array(integrals)
    s3_indx = np.where(np.diff(np.sign(integrals - 0.9973)))[0]
    s3_height = heights[s3_indx][0]
    
    return [s3_height, s2_height, s1_height]

# %%

# test the above
if __name__ == '__main__':
    # define 2D grids
    v = np.linspace(0, 1, 2000)
    xg, yg = np.meshgrid(v, v)
    
    # define a 2D unnormalised gaussian:
    F = xg*yg**2
    
    # normalise it to Volume = 1: with our integrate 2D fucntion:
    norm = integrate2D(F, xg, yg)
    F *= 1/norm
    
    # check that it's right by analytical result
    print(f'For F(x, y)= x*y^2 in [0, 1]^2 we expect volume of  {np.round(1/6, 4)}')
    print('\n')
    print(f'Our intgeration fucntion gave: {np.round(norm, 4)}')
    
    # give it to the confidence fucntion to get sigma regions:
    levels = confidence(F, xg, yg, accu=3000)
    
    # plot these as contours on a conour map
    fig = plt.figure()
    ax = fig.gca()
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)
    heatmap = ax.pcolormesh(xg, yg, F)
    contourplot = ax.contour(xg, yg, F, levels, cmap=cm.jet)
    ax.clabel(contourplot, fontsize=16)
    fig.colorbar(heatmap)


# %%

# extra convergence of 2D integration
if __name__ == '__main__':
    
    # define array of accuracies:
    accu = np.linspace(10, 200, 10)
    
    # loop over each finding norms:
    norms = []
    for a in accu:
        # define starting 2D grids
        v = np.linspace(0, 1, a)
        xg, yg = np.meshgrid(v, v)
        
        # define a 2D unnormalised gaussian:
        F = xg*yg**2
        
        # normalise it to Volume = 1: with our integrate 2D fucntion:
        norm = integrate2D(F, xg, yg)
        F *= 1/norm
        # save it
        norms.append(norm)
    
    # find error:
    change = abs((np.diff(norms) / norms[:-1])*100)
    
    # plot:
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$grid \ accuracy$', fontsize=20)
    ax1.set_ylabel(r'$relative \ error \ [\%]$', fontsize=20)
    ax1.plot(accu[1:], change)
    
    # plot horizontal levels for reference
    ax1.axhline(1, color='r', ls='-.', label=r'$1\%$')
    ax1.axhline(0.5, color='g', ls='-.', label=r'$0.5\%$')
    ax1.axhline(0.1, color='b', ls='-.', label=r'$0.1\%$')
    ax1.axhline(0.01, color='orange', ls='-.', label=r'$0.01\%$')
    
    ax1.legend(fontsize=20)

# %%

# extra convergence study of confidence heights and their time to run

import time

if __name__ == '__main__':
    
    # set up h accuracies to use:
    accus = np.arange(100, 3000, 300)
    values = []  # list to store results for 1 sigma
    times = []  # list to save time to run for each
    
    for a in accus:
        
        start = time.perf_counter()
        
        # define 2D grids
        v = np.linspace(0, 1, 1000)
        xg, yg = np.meshgrid(v, v)
        
        # define a 2D unnormalised gaussian:
        F = xg*yg**2
        
        # normalise it to Volume = 1: with our integrate 2D fucntion:
        norm = integrate2D(F, xg, yg)
        F *= 1/norm
        
        # give it to the confidence fucntion to get sigma regions: and save 1sigma one
        levels = confidence(F, xg, yg, accu=a)
        values.append(levels[2])
        
        # save time to run
        end = time.perf_counter()
        times.append(end-start)
        
        # inform progress:
        print(f'done run with accuracy = {a}')
    
    # find relative errors
    values = np.array(values)
    change = abs((np.diff(values) / values[:-1])*100)
    
    # plot them:
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$height \ accuracy$', fontsize=20)
    ax1.set_ylabel(r'$relative \ error \ [\%]$', fontsize=20)
    ax1.plot(accus[1:], change)
    
    # plot horizontal levels for reference
    ax1.axhline(1, color='r', ls='-.', label=r'$1\%$')
    ax1.axhline(0.5, color='g', ls='-.', label=r'$0.5\%$')
    ax1.axhline(0.1, color='b', ls='-.', label=r'$0.1\%$')
    ax1.axhline(0.01, color='orange', ls='-.', label=r'$0.01\%$')
    
    ax1.legend(fontsize=20)
    
    # plot the last one as contours on a conour map
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x$', fontsize=20)
    ax2.set_ylabel(r'$y$', fontsize=20)
    heatmap = ax2.pcolormesh(xg, yg, F)
    contourplot = ax2.contour(xg, yg, F, levels, cmap=cm.jet)
    ax2.clabel(contourplot, fontsize=16)
    fig2.colorbar(heatmap)
    
    # plot the time to run for each
    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.tick_params(labelsize=16)
    ax3.set_xlabel(r'$height \ accuracy$', fontsize=20)
    ax3.set_ylabel(r'$time \ to \ run \ [s]$', fontsize=20)
    ax3.plot(accus, times)








