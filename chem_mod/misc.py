import numpy as np
import matplotlib.pyplot as plt

def make_legend_axes(ax,pad):
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes('right',0.15, pad=pad)
    return legend_ax

def contour_points(x,y,z,nx,ny,ax=None,log=True,vmin=None,vmax=None,levels=25,**kwargs):
    '''
    General function for plotting contour maps given x, y, and z points.
        ARGUMENTS:
            x,y    - Lists of positions of each point to make contour from.
            z      - List of heights of the surface at the corresponding x,y positions.
            ax     - A matplotlib.axes subplot object on which to draw the contour map.
                     If one is not provided, it will be created.
            vmin   - Minimum z-value to distinguish in colormap.
            vmax   - Maximum z-value to distinguish in colormap.
            levels - Array or int specifying contour levels.
            cmap   - Colormap to use.
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
    sortx = np.argsort(X[:,0])
    Y = np.array(y).reshape((nx,ny))
    Z = np.array(z).reshape((nx,ny))
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

def remove_nan(x,y,z):
    '''
    Generate a mask that gets rid of x rows and y columns that are entirely nan in z.
    '''
    mask = np.zeros_like(x,dtype=bool)
    xvals = list(set(x))
    for xval in xvals:
        if np.any(np.isnan(z[x == xval])):
            mask = mask | (x == xval)
    yvals = list(set(y))
    for yval in yvals:
        if np.all(np.isnan(z[y == yval])):
            mask = mask | (y == yval)
    return ~mask

def sigfig(x,n):
    om = 10**np.round(np.log10(x))
    xr = np.round(x/om,n-1)*om
    return xr

def iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False