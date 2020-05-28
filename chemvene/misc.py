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
    cbar_ax.minorticks_off()
    cbar_ax.tick_params(axis='y',which='major',direction='out')
    return cbar_ax

def get_contour_arr(x,nx,ny,sortx=None):
    X = np.array(x).reshape((nx,ny))
    if not sortx is None:
        sortX = np.array(sortx).reshape((nx,ny))
        sort = np.argsort(sortX[:,0])
        X = X[sort,:]
    return X

def contour_points(x,y,z,nx,ny,ax=None,log=True,vmin=None,vmax=None,levels=25,fill=True,cbar=True,return_artist=False,**kwargs):
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
            cbar   - Boolean whether or not to display the colormap.
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
        if log:
            levels = np.log10(levels)
    except TypeError:
        levels = np.linspace(vmin,vmax,levels)
        ticks  = np.linspace(vmin,vmax,10)
    X = get_contour_arr(x,nx,ny,sortx=x)
    Y = get_contour_arr(y,nx,ny,sortx=x)
    Z = get_contour_arr(z,nx,ny,sortx=x)
    if fill:
        cont = ax.contourf(X,Y,Z,levels=levels,vmin=vmin,vmax=vmax,extend='both',**kwargs)
    else:
        if 'colors' in kwargs and kwargs['colors'] != None and 'cmap' in kwargs:
            kwargs = dict(kwargs)
            kwargs['cmap'] = None
        cont = ax.contour(X,Y,Z,levels=levels,vmin=vmin,vmax=vmax,extend='both',**kwargs)
    if fill and cbar:
        cax = None # make_legend_axes(ax,pad=0.1)
        if log:
            cbar = plt.colorbar(cont,cax=cax,ax=ax,ticks=ticks,format=r'$10^{%4.1f}$')
        else:
            cbar = plt.colorbar(cont,cax=cax,ax=ax,ticks=ticks,format=r'$%4.2f$')

    if return_artist:
        return ax,cont
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
        x  - scalar or array of numbers that can be cast to np.float32.

    RETURNS:
        xr - scalar or array of rounded numbers, with n significant figures.
    '''
    try:
        x = np.array(x,dtype=np.float32)
        assert len(x.shape) > 0
    except AssertionError:
        x = np.float32(x)
    om = 10**np.floor(np.log10(x/np.sign(x)))
    xr = np.round((x)/om,n-1)*om
    return xr

def iterable(x,count_str=False):
    '''
    Function for determining if a variable is iterable or not.

    ARGUMENTS:
        x         - The variable you want to check
        count_str - Boolean whether or not strings should count as iterable.
                    If type(x) = str and count_str = True,  then iterable(x) returns True.
                                      if count_str = False, then iterable(x) returns False.
    RETURNS:
        Boolean True is x is iterable and False if x is scalar. 
    '''
    try:
        iter(x)
        if not count_str:
            assert type(x) != str
        return True
    except (TypeError,AssertionError) as e:
        return False

def nint(x,y):
    '''
    Function for numerically integrating a simple 1D function.
    Integration is performed using the trapezoidal method.
    '''
    if np.all(np.isnan(y)):
        return np.nan
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    S = 0
    S = np.nansum([ (ylo+yhi)/2 * (xhi-xlo) for xlo,xhi,ylo,yhi in zip(x[:-1],x[1:],y[:-1],y[1:])])
    return S
