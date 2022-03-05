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


def confidence(F, xg, yg, accu, interp, interp_kind='linear'):
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
