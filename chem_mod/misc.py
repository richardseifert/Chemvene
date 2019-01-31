import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_legend_axes(ax,pad=0.15):
    '''
    Function for creating a colorbar axes next to a plotting axes.

    ARGUMENTS:
        ax      - The plotting axes you want a colorbar for.
        pad     - The amount of space between plotting axes and colorbar axes. Default pad=0.15.
    RETURNS:
        cbar_ax - The axes object to plot a colorbar in.
    '''
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('right',0.15, pad=pad)
    return cbar_ax

def contour_points(x,y,z,nx,ny,ax=None,log=True,vmin=None,vmax=None,levels=25,**kwargs):
    '''
    General function for plotting contour maps given x, y, and z points.
        ARGUMENTS:
            x,y    - Lists of positions of each point to make contour from.
            z      - List of heights of the surface at the corresponding x,y positions.
            nx,ny  - 2D dimensions so that x,y,z can all be reshaped into 2D arrays.
            ax     - A matplotlib.axes subplot object on which to draw the contour map.
                     If one is not provided, it will be created.
            vmin   - Minimum z-value to distinguish in colormap. All values below this are
                     plotted the same color.
            vmax   - Maximum z-value to distinguish in colormap. All values above this are
                     plotted the same color.
            levels - Array or int specifying contour levels.
            kwargs - Any named arguments accepted by matplotlib.pyplot.contourf can be provided
                     and will be passed to contourf when it is called.
                     Common examples are: cmap, hatches, etc.
        RETURNS
            ax     - A matplotlib.axes subplot object with the contour map drawn.
    '''
    if ax is None:
        fig,ax = plt.subplots()
    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    if log:
        z = np.log10(z+1e-30)
        vmin = np.log10(vmin+1e-30)
        vmax = np.log10(vmax)
    try:
        iter(levels)
    except TypeError:
        levels = np.linspace(vmin,vmax,levels)
        ticks  = np.linspace(vmin,vmax,10)
    print(ticks)
    X = np.array(x).reshape((nx,ny))
    Y = np.array(y).reshape((nx,ny))
    Z = np.array(z).reshape((nx,ny))
    sortx = np.argsort(X[:,0])
    X = X[sortx,:]
    Y = Y[sortx,:]
    Z = Z[sortx,:]
    cont = ax.contourf(X,Y,Z,levels=levels,vmin=vmin,vmax=vmax,extend='both',**kwargs)
    cax = make_legend_axes(ax,pad=0.1)
    if log:
        cbar = plt.colorbar(cont,cax=cax,ticks=ticks,format=r'$10^{%4.1f}$')
    else:
        cbar = plt.colorbar(cont,cax=cax,ticks=ticks,format=r'$%4.2f$')
    return ax

def remove_nan(x,z):
    '''
    Generate a mask that gets rid of rows that are entirely nan in z.

    ARGUMENTS:
    x - x-values corresponding to the z-values given.
    z - z-values.
    '''
    mask = np.zeros_like(x,dtype=bool)
    xvals = list(set(x))
    for xval in xvals:
        if np.any(np.isnan(z[x == xval])):
            mask = mask | (x == xval)
    return ~mask

def sigfig(x,n):
    '''
    Take a number or array of numbers and round them 
    to a specified number of significant digits.

    ARGUMENTS:
    x - scalar or array of 
    '''
    try:
        x = np.array(x)
        assert len(x.shape) > 0
    except AssertionError:
        x = np.float32(x)
    om = 10**np.floor(np.log10(x/np.sign(x)))
    print(om)
    xr = np.round((x)/om,n-1)*om
    print(om)
    return xr

def iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False