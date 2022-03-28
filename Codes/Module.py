'''
Contains functions we continiously use
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
    s1_indx = np.where(np.diff(np.sign(integrals - 0.683)))[0]
    s1_height = heights[s1_indx][0]
    
    # 2-sigma:
    integrals = np.array(integrals)
    s2_indx = np.where(np.diff(np.sign(integrals - 0.955)))[0]
    s2_height = heights[s2_indx][0]
    
    # 3-sigma:
    integrals = np.array(integrals)
    s3_indx = np.where(np.diff(np.sign(integrals - 0.997)))[0]
    s3_height = heights[s3_indx][0]
    
    return [s3_height, s2_height, s1_height]


# function to plot confidence intervals on a 1D chi^2
def chi_confidence1D(y, x, axis, colors=['r', 'g', 'b'], labels=True, interp=10000):
    '''
    Finds and plots default confidence interval lines for 1D chi^2
    The inuput y axis must be chi^2, reduced to minimum at y=0.
    
    Prameters:
        ---------------
        - y - numpy.ndarray y values
        - x - numpy.ndarray x values
        - indx1 - index of intersection with line at 1 sigma
        - indx2 - index of intersection with line at 2 sigma
        - indx3 - index of intersection with line at 3 sigma
        - ax - matplotlib axis to plot on
        - colors - colours to use for lines at the 3 intervals, in order
                   default = ['r', 'g', 'b']
        - labels - bool - default = True, defines if legend is to draw
        -- interp - int, if > 10, we interpolate grids and F to that size,
                 default = 0
    Returns:
        1 sigma x values as list in format [-error, +error]
    '''
    
    # interpolate if user wished to
    if interp > 10:
        y = np.interp(np.linspace(x[0], x[-1], interp), x, y)
        x = np.linspace(x[0], x[-1], interp)
    
    # get indexes of sigma level intersections
    indx1 = np.argwhere(np.diff(np.sign(y - np.ones(np.shape(y)))))
    indx2 = np.argwhere(np.diff(np.sign(y - 2.71*np.ones(np.shape(y)))))
    indx3 = np.argwhere(np.diff(np.sign(y - 9*np.ones(np.shape(y)))))
    
    # get minimum and so mean x measurement:
    indx_min = np.where(y == np.min(y))
    x_mean = x[indx_min]
        
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
    
    # return values of 1 sigma regions
    return x_mean[0] - x1[0], x1[1] - x_mean[0]


# example of chi_confidence1D use:
if __name__ == '__main__':
    x = np.linspace(-2, 4, 100)
    
    chi = 5*(x-2)**2
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\chi^2$')
    ax.plot(x, chi)
    
    chi_confidence1D(chi, x, ax)


# %%
