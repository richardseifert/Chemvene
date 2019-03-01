import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pandas import DataFrame
import glob
import os

#Package Imports
from .read.read_abunds import find_mol, load_mol_abund
from .read.read_rates import load_rates, get_reac_str, total_rates
from .misc import contour_points, remove_nan, sigfig, iterable, nint
from chem_mod import __path__ as pkg_path

#Path to the Chemical Code Directory.
#bsd = '/bucket/ras8qnr/MasterChem_Phobos/'
bsd = '/home/ras8qnr/MasterChem_Rivanna/'

#Some constants that get used throughout.
mp = 1.67e-24  #Mass of proton in g
mau = 1.496e11 #Conversion from AU to meters.

class chem_mod:
    '''
        A class to handle loading, viewing, and manipulating output from
        the disk chemical modeling code presented in Fogel et al. 2011.
        For more in-depth documentation, visit 

                https://github.com/richardseifert/chem_mod

        To create an instance, the following three paths must be provided.
            environ - string path to the environ/ directory used to run your
                      chemical model.
            inp     - string filename of the input file used to run your model.
            outdir  - string path to the runs/ directory where model output is
                      stored.
    '''

    ################################################################################
    ################################ Initialization ################################
    ################################################################################

    def __init__(self,environ,inp,outdir,bsd=bsd):
        self.set_environ(environ)
        self.set_inp(inp)
        self.outdir = outdir
        if self.outdir[-1] != '/':
            self.outdir += '/'
        self.phys = DataFrame()
        self.abunds = {}
        self.rates  = {}

        self.load_physical()
        self.load_times()
    def set_environ(self,environ):
        self.environ = environ
        if self.environ[-1] != '/':
            self.environ += '/' 
    def set_inp(self,inp):
        self.inp = self.environ+inp
        self.inp_paths = {k:None for k in ['spec','reac','uv','xray','isrf','rn']}
        d = np.genfromtxt(self.inp,dtype=str)
        for i,k in enumerate(self.inp_paths.keys()):
            if os.path.exists(bsd+d[i]):
                self.inp_paths[k] = bsd+d[i]

    ################################################################################
    ############################### General Loading ################################
    ################################################################################

    def merge(self,tbl):
        '''
        Prepare a given table to be merged according to position, R and zAU.

            ARGUMENTS:
              tbl -    A pandas table containing the two columns 'R' and either 'shell' or 'zAU'.

            RETURNS:
              merged - A tbl with the same number of rows as phys. The returned table
                       has values ordered according to phys['R'] and phys['shell']
        '''

        #Match R values to their nearest R values in phys['R'].
        #This is necessary for the relational merge to work.
        phys_R = np.array(list(set(self.phys['R'])))
        diffs = np.vstack([(pr-tbl['R'])**2 for pr in phys_R])
        inds = np.argmin(diffs,axis=0)
        tbl['R'] = phys_R[inds]

        #Merge according to columns of phys.
        if 'shell' in tbl.columns:
            merged = self.phys.merge(tbl,'left',on=['R','shell'])
        elif 'zAU' in tbl.columns:
            #Match zAU values at each radius to phys['zAU']
            for R in phys_R:
                phys_mask = self.phys['R'] == R
                tbl_mask = tbl['R'] == R
                phys_z = np.array(self.phys['zAU'][phys_mask])
                diffs = np.vstack([(pz-tbl['zAU'][tbl_mask])**2 for pz in phys_z])
                inds = np.argmin(diffs,axis=0)
                tbl['zAU'][tbl_mask] = phys_z[inds]
            merged = self.phys.merge(tbl,'left',on=['R','zAU'])
        return merged
    
    def set_times(self,tbl):
        '''
        Method that takes a table with times as column headers and changes the headers
        to match the nearest model timesteps.

        ARGUMENTS:
            tbl - A pandas.DataFrame object with times (in years) as columns header.
        RETURNS:
            The same table, but times have been corrected to the nearest model times.
        '''
        ctimes = tbl.columns
        mtimes = self.nearest_times(ctimes,itr=True)
        return tbl.rename(columns=dict(zip(ctimes,mtimes)))

    ################################################################################
    ########################## Handling Model Timesteps ############################
    ################################################################################

    def load_times(self):
        '''
        Method that reads the 2times.inp file for the model and produces an array of
        the time at each model timestep.

        No arguments or returns; times are stored in self.times variable.
        '''
        f = open(self.outdir+'2times.inp')
        f.readline()
        t_end = float(f.readline().split()[0].replace('D','E'))
        t_start = float(f.readline().split()[0].replace('D','E'))
        nsteps = float(f.readline().split()[0])
        self.times = sigfig(np.logspace(np.log10(t_start),np.log10(t_end),nsteps), 4)

    def nearest_times(self,times,itr=False):
        '''
        Function for finding nearest timesteps to a given time or list of times.

        ARGUMENTS:
            times - Time or list of times. Must be values that can be cast to floats.
            itr   - Boolean whether or not to return a scalar if possible. Default False.
                    If a single time is given, itr=False will return a scalar value.
                                               itr=True  will return a list of length one.
        '''
        #If None was given, do nothing. Return None.
        if times is None:
            return times
        
        #Otherwise, check if time is iterable. If it's not, make it a single-valued array.
        try:
            iter(times)
            times = np.array(times).astype(float)
        except TypeError:
            times = np.asarray([times]).astype(float)
        
        #Find the nearest times in self.times.
        nearest = self.times[ np.argmin([ (self.times - t)**2 for t in times ], axis=1) ]

        #Depending on the number of times given, return an array or a scalar.
        if len(nearest) == 1 and not itr:
            return nearest[0]
        else:
            return nearest

    ################################################################################
    ########################## Handling Physical Model #############################
    ################################################################################

    def load_physical(self):
        '''
        Method that loads the disk physical model from 1environ files.

        No arguments or returns; physical model is stored in a pandas.DataFrame
        object, self.phys
        '''
        env_paths = glob.glob(self.environ+'1environ*')

        dat = np.array([])
        shells = np.array([np.arange(1,51)]).T
        for path in env_paths:
            d = np.loadtxt(path,skiprows=3)
            d = np.hstack([d,shells])
            if len(dat) != 0:
                dat = np.vstack([dat,d]) 
            else:
                dat = d
        
        #Get header from test file.
        f = open(env_paths[0])
        header = f.readline()
        f.close()

        for i,k in enumerate(header.split()+['shell']):
            self.phys[k] = dat[:,i]

    ################################################################################
    ####################### Handling Abundances of Species #########################
    ################################################################################

    def limedir(self,strmol):
        '''
        Function that produces string limefg path for a given species.
        It's a pretty pointless method, because I only need the limefg path
        twice, when loading and writing species abundances. But, I figured
        if I ever want to change where I save limefg or what I want to name
        the directory, I can just change it once in this method.

        ARGUMENTS:
            strmol - String name of the species.

        RETURNS:
            string path of a directory where limefg should go.
        '''
        return self.outdir+'e1/limefg_'+strmol+'/'

    def load_mol(self,strmol,times=None):
        '''
        Method that loads abundances of a given species, 
        potentially at a given time or times.

        If limefg exists for this species (it has previously been loaded and saved),
        then species if loaded from this (quicker). Otherwise, species is loaded
        directly from r*.out files.

        ARGUMENTS:
            strmol - string name of the species to load.
            times  - Time steps to load species at. Only works if species is saved
                     in limefg format. Optional; default times=None -> load all times.
        RETURNS:
            Nothing, abundances are stored in self.abunds[strmol]. Column headers are
            model times. Use self.get_quant to get strmol at a specific time (See below).
        '''
        #Look for strmol in limefg format.
        limedir = self.limedir(strmol)
        if not os.path.exists(limedir):
            #If not in limefg, load from scratch (and write to limefg).
            self.read_mol(strmol,write=True)
            return

        #Load from limefg
        print("Loading from limefg.")
        self.abunds[strmol] = DataFrame()
        outpaths = glob.glob(self.outdir+'e1/r*.out')
        
        limepaths = glob.glob(limedir+'*time*.dat')
        tnum = [int(lp.split('time')[-1].split('.')[0]) for lp in limepaths]
        limepaths = np.array(limepaths)[np.argsort(tnum)]

        #Only load files for times requested.
        all_times = self.times
        times = self.nearest_times(times,itr=True)
        limepaths = [ lp for t,lp in zip(all_times,limepaths) if t in times ]
        
        abunds = np.array([])
        columns = ['R','zAU','rho','Tgas','Tdust','abund','fg']
        for time,path in zip(times,limepaths):
            dat = np.loadtxt(path)
            tbl = DataFrame(dat,columns=columns)
            tbl['R'] /= mau
            tbl['zAU'] /= mau
            merged = self.merge(tbl)
            self.abunds[strmol][time] = merged['abund']
        #Tweak times to be exact values from self.times.
        self.abunds[strmol] = self.set_times(self.abunds[strmol])

    def read_mol(self,strmol,write=False):
        '''
        Method that reads abundances of a given species from r*.out files.

        ARGUMENTS:
            strmol - string name of the species to load.

        RETURNS:
            Nothing, abundances are stored in self.abunds[strmol]. Column headers are
            model times. Use self.get_quant to get strmol at a specific time (See below).
        '''
        #Load from e1 files.
        dat = load_mol_abund(self.outdir+'e1/',strmol)
        times = list(set(dat[:,0]))
        t = dat[:,0]
        R = dat[:,1]
        shell = dat[:,2]
        abunds = dat[:,3]

        #Construct table with abundances at each timestep.
        mol_abund = DataFrame({time:abunds[t==time] for time in sorted(times)})
        mol_abund['shell'] = shell[t==times[0]]
        mol_abund['R'] = R[t==times[0]]

        #Merge table with existing self.phys physical table.
        self.abunds[strmol] = self.merge(mol_abund)[times]

        #Tweak times to be exact values from self.times.
        self.abunds[strmol] = self.set_times(self.abunds[strmol])

        if write:
            #Write abundances in limefg format.
            self.write_mol(strmol)

    def write_mol(self,strmol):
        '''
        Method that writes abundances for a species in the limefg format
        used by LIME radiative transfer.

        ARGUMENTS:
            strmol - string name of the species to load.
        '''
        if not strmol in self.abunds.keys():
            self.read_mol(strmol)
        savetbl = self.phys[['R','zAU','rho','Tgas','Tdust']]
        savetbl.loc[:,'rho'] *= 0.8/(2.0*mp) * 1e6
        savetbl.loc[:,'abund'] = np.zeros_like(savetbl['R']) #Place holder.

        # Match tmp table and physical table by positions.
        tmp = np.genfromtxt(pkg_path[0]+'/pkg_files/imlup_gaia_v2_abrig_model_Tgas_SB_G04.txt')     
        inds = [np.argmin(( tmp[:,0]-R)**2 + (tmp[:,1]-z)**2 ) for R,z in zip(self.phys['R'],self.phys['zAU'])]
        tmp_sort = tmp[inds]
        fghere = np.array(tmp_sort[:,2]/(tmp_sort[:,3]*tmp_sort[:,7]))
        fghere[(tmp_sort[:,3] <= 1e-30) | (tmp_sort[:,7] <= 1e-30)] = 1e20
        fghere[savetbl['R'] > 313.] = 1e20 # this is for IM LUP SPECIFICALLY!! no large grains beyond this radius
        savetbl.loc[:,'fg'] = fghere

        savetbl.loc[:,'R'] *= mau
        savetbl.loc[:,'zAU'] *= mau

        limedir = self.limedir(strmol)
        if not os.path.exists(limedir):
            os.makedirs(limedir)
        times = list(set(self.abunds[strmol].columns))
        for i,time in enumerate(times):
            fname=limedir+strmol+'_time'+str(i)+'.dat'
            abu = np.array(self.abunds[strmol][time])

            abu[(savetbl['rho'] <= 1e4) | (abu < 1e-28)] = 0.0
            savetbl.loc[:,'abund'] = abu
            no_nan = remove_nan(self.phys['R'],abu)
            savearr = np.array(savetbl)[no_nan]
            np.savetxt(fname,savearr,fmt='%15.7E')

    ################################################################################
    ######################### Handling Species Reactions ###########################
    ################################################################################
    
    def get_reac_str(self,reac_id,fmt='ascii'):
        '''
        Method that obtains a string representation of a given reaction in the
        chemical network.

        ARGUMENTS:
            reac_id - Integer ID for the reaction.
            fmt     - Desired format of the reaction string.
                      Options:
                        ascii - Plain text, no subscript or superscript.
                        latex - Formatted to include subscripts and superscripts
                                when interpreted by LaTeX.
        RETURNS:
            Reaction string.
        '''
        return get_reac_str(self.inp_paths['reac'], reac_id, fmt)

    def load_reac(self,strmol,reacs,times=None,radii=None):
        '''
        Method for loading reaction rates for a specific reaction or reactions
        involving a specific species, optionally at specific times or radii.

        ARGUMENTS:
            strmol - Species involved in the reaction(s). This is used as the prefix
                     for the *.rout files that contain reaction rates.
            reacs  - Scalar or array of integer reaction IDs.
            times  - Model timesteps at which to load reaction rates. Default is all times.
            radii  - Model radii at which to load reaction rates. Default is all radii.
        RETURNS:
            Nothing, rates are stored in self.rates[reac_id]. Column headers are
            model times. Use self.get_quant to get rates at a specific time (See below).
        '''
        #Check that this molecule has reaction rates collated.
        reac_files = glob.glob(self.outdir+'e1/rates/'+strmol+'_*.rout')
        if len(reac_files) == 0:
            print("Warning: This molecule has no reaction rates stored for %s. \
                   Doing nothing and continuing.")
        #Find nearest times to those given.
        if not times is None:
            times = self.nearest_times(times)
        #Load from e1 files.
        dat = load_rates(self.outdir+'e1/rates/',strmol,reacs,times,radii)

        times = list(set(dat[:,0]))
        t = dat[:,0]
        R = dat[:,1]
        shell = dat[:,2]
        reac_ids = dat[:,3]
        reac_rates = dat[:,4]

        try:
            iter(reacs)
        except TypeError:
            reacs = [ reacs ]
        for reac in reacs:
            self.rates[reac] = DataFrame()
            for time in times:
                #Construct table with abundances at each timestep.
                mask = (reac_ids==reac) & (t==time)
                tbl = DataFrame()
                tbl[time] = reac_rates[mask]
                tbl['shell'] = shell[mask]
                tbl['R'] = R[mask]
                self.rates[reac][time] = self.merge(tbl)[time]

    def rank_reacs(self,strmol,time=None,R=None):
        '''
        Method for ranking reactions involving a particular species
        according to the reaction rates, optionally at a specific time
        and/or radius in the model.

        ARGUMENTS:
            strmol - The species whose reactions will be ranked.
            time   - Timestep at which to rank reactions.
                     Default, sum over all timesteps.
            R      - Radius at which to rank reactions.
                     Default, sum over all radii.
        '''
        if not time is None:
            time = self.nearest_times(time)
        rates = total_rates(self.outdir+'e1/rates/',strmol,time,R)
        return rates

    ################################################################################
    ############################# Requesting Model Data ############################
    ################################################################################

    def get_quant(self,quant,time=0):
        '''
        Method for obtaining model quantity at all locations of the disk,
        at a specific time.

        ARGUMENTS:
            quant - Name of quantity. String for physical quantities and species
                    abundances. Integer for reaction IDs.
                      For convenience of other methods that use get_quant, if an array
                       of values is passed, get_quant will do nothing and return the 
                       array passed to it.
            time  - Float value of the time at which to get the quantity.
        RETURNS:
            1D array of quant values corresponding to R and shell/zAU columns of self.phys
            ## Eventually I'm going to add an option to reshape the output into a 2D array
               with constant R columns and constant shell rows. ##
        '''
        if iterable(quant):
            pass #quant is already 2D values.
        elif quant in self.phys.columns:
            quant = self.phys[quant]
            #("Found quant in physical.")
        elif quant in self.abunds.keys():
            times = np.array(self.abunds[quant].columns)
            #nearest = self.nearest_times(times)    #Times won't necessarily align with abunds columns. Figure that out first.
            nearest = times[np.argmin((times-time)**2)]
            quant = self.abunds[quant][nearest]
            #print("Found quant in abundances.")
        elif quant in self.rates.keys():
            times = np.array(self.rates[quant].columns)
            #nearest = self.nearest_times(times)    #Times won't necessarily align with rates columns. Figure that out first.
            nearest = times[np.argmin((times-time)**2)]
            quant = self.rates[quant][nearest]
            if np.nanmean(quant) < 0:
                quant = -quant
            #print("Found quant in rates.")
        elif quant[0]=='n' and quant[1:] in self.abunds.keys():
            quant = self.abs_abund(quant[1:],time)
        else:
            raise ValueError("The quantity %s was not found for this model."%(quant))
        return quant
    
    def z_quant(self,quant,R=100,time=0):
        '''
        Method for obtaining quant as a function of Z at a particular radius and time.

        ARGUMENTS:
            quant - The quantity you're interested in. Could be physical quantity,
                    chemical species for abundances, or reaction ID for rates.
            R     - Radius at which to return quant. Default is R = 100 AU.
            time  - Time at which to return quant. Defaults to first timestep.
        Returns
            z     - 1D heights in AU.
            quant - 1D quant values corresponding to z.
        '''
        #Copy quant string before it's overwritten.
        quant_str = (str(quant)+'.')[:-1]

        #Find nearest R value in grid.
        radii = np.array(list(set(self.phys['R'])))
        R = radii[np.argmin(np.abs(radii-R))]

        #Get 1-D arrays of z and quant at specified R value.
        R_mask = self.phys['R'] == R
        z = np.array(self.phys['zAU'][ R_mask ])
        quant = self.get_quant(quant,time)
        quant = np.array(quant[ R_mask ])

        #Sort by z
        sort = np.argsort(z)
        z = z[sort]
        quant = quant[sort]

        return z,quant
    
    def R_quant(self,quant,shell=0,ax=None):
        '''
        Method for obtaining quant as a function of radius at a particular shell and time.

        ARGUMENTS:
            quant - The quantity you're interested in. Could be physical quantity,
                    chemical species for abundances, or reaction ID for rates.
            shell - Shell at which to return quant. Default is shell = 0, the
                    outer layer of the disk.
            time  - Time at which to return quant. Defaults to first timestep.
        Returns
            R     - 1D radii in AU.
            quant - 1D quant values corresponding to R.
        '''
        R = self.phys['R']
        quant = self.get_quant(quant)
        mask = self.phys['shell'] == shell
        R,quant = R[mask],quant[mask]
        sort = np.argsort(R)
        R,quant = R[sort],quant[sort]
        return R,quant

    def abs_abund(self, strmol, time=0):
        ab = self.get_quant(strmol,time=time) # per number density Hydrogen nuclei.
        rho = self.get_quant('rho')
        nH = rho / mp
        nX = np.array(ab*nH)
        return nX

    def get_cd_vertical(self,mol,time=0):
        #Copy inputs, so they aren't changed.
        R = np.array(self.get_quant('R')) * mau*100
        Z = np.array(self.get_quant('zAU')) * mau*100
        if not mol in self.abunds.keys():
            self.load_mol(mol,times=time)
        nX = np.array(self.get_quant('n'+mol))
        N = np.zeros_like(R)

        #Sort inputs into rows
        sort = np.lexsort((-Z,R))
        unsort = np.argsort(sort)
        R = R[sort]
        Z = Z[sort]
        nX = nX[sort]

        for i in range(1,len(R)):
            if R[i] != R[i-1]:
                N[i] = 0.0
            else:
                N[i] = N[i-1] + 0.5*(nX[i]+nX[i-1])*(Z[i-1]-Z[i])

        return N[unsort]
    def get_cd_radial(self,mol,time=0):
        pass

    def get_shield_fi(self,mol,alpha,delta,zeta,cd_mode='vertical',time=0):
        if cd_mode == 'vertical':
            N = self.get_cd_vertical(mol,time=time)
        elif cd_mode == 'radial':
            N = self.get_cd_radial(mol,time=time)
        else:
            raise ValueError("Invalid column density mode %s"%(cd_mode))

        x = N/zeta
        f = (1+x)**-delta * np.exp(-alpha*x)
        return f
    
    def column_density(self,strmol,time=0):
        '''
        Method for producing columnd density profile for a given species.

        ARGUMENTS:
            strmol - string of the molecule you want to get column density of.
            time   - timestep you want columnd density at.
        RETURNS:
            R_vals - Radius values.
            cd     - Corresponding column densities at those radii.
        '''
        #Load number density of strmol (cm^-3).
        ab = self.get_quant(strmol,time=time) # per number density Hydrogen nuclei.
        rho = self.get_quant('rho')
        nH = rho / mp 
        nX = np.array(ab*nH)
        self.profile_quant(nX)

        #Load corresponding disk locations.
        R = np.array(self.get_quant('R'))
        R_vals = np.unique(R) #Get unique values of R.
        R_vals = R_vals[np.argsort(R_vals)]
        Z = np.array(self.get_quant('zAU'))

        #At each radius, numerically integrate number density over the disk height
        # to get column density in cm^-2
        cd = np.zeros_like(R_vals)
        for i,r in enumerate(R_vals):
            at_R = R == r
            n = nX[at_R]
            z = Z[at_R]
            z = z*mau * 100 #Convert from AU to cm
            cd[i] = 2*nint(z,n) #The 2 is to account for both halves of the disk.
        
        return R_vals,cd


    ################################################################################
    ################################## Plotting ####################################
    ################################################################################

    def profile_quant(self,quant,time=0,vmin=None,vmax=None,plot_grid=False,**kwargs):
        '''
        Method for plotting disk profile in a specified quantity (e.g. Dust temperature, HCO+ abundance, etc.).

        ARGUMENTS:
            quant     - The quantity you want to see a disk profile of.
            time      - The timestep at which to produce the profile. Defaults to first timestep.
            log       - Plot profile on logspace colormap. Defaults to True.
            ax        - matplotlib.pyplot.axes object to plot profile onto. Default, make a new one.
            vmin,vmax - Colormap upper and lower bounds. By default, they are determined from the
                        minimum and maximum values of the quantity you're plotting.
            levels    - Number of contour levels to use, or array of contour values.
            plot_grid - Boolean whether or not to plot gridpoints on top of contours. Defaults to False.

        RETURNS:
            ax        - The axes object with the contours plotted.
        '''
        quant = self.get_quant(quant,time)
        R = self.phys['R']
        z = self.phys['zAU']
        if vmin is None:
            vmin = np.nanmin(quant[quant>0])
        if vmax is None:
            vmax = np.nanmax(quant[quant>0])
        nx = len(list(set(self.phys['R'])))
        ny = len(list(set(self.phys['shell'])))
        ax = contour_points(R,z,quant,nx=nx,ny=ny,vmin=vmin,vmax=vmax,**kwargs)

        if plot_grid:
            ax.scatter(R,z,s=1,color='black')
        ax.set_xlabel('R (AU)')
        ax.set_ylabel('Z (AU)')
        return ax

    def profile_reac(self,reac,time=0,**kwargs):
        '''
        Method for plotting disk profile in the rate of a specific reaction.
        Same as profile_reac above, but it grabs the reaction string to use
        as a title.
        '''
        ax = self.profile_quant(reac,time=time,**kwargs)
        ax.set_title( self.get_reac_str(reac,fmt='latex') )
    
    def profile_best_reacs(self,strmol,n,time=None,rank_R=None,**kwargs):
        rates = self.rank_reacs(strmol,time,rank_R)
        rates = rates[:n]

        for rid,rate in rates:
            print("Loading "+self.get_reac_str(rid))
            self.load_reac(strmol,rid,times=time)
            if not 'cmap' in kwargs:
                if rate > 0:
                    cmap = 'Blues'
                else:
                    cmap = 'Reds'
                self.profile_reac(rid,time=time,cmap=cmap,**kwargs)
            else:
                self.profile_reac(rid,time=time,**kwargs)

    def z_best_reacs(self,strmol,n,R,time=None,plot_mols=None,total=True,cmap_pro='Blues',cmap_des='Reds',load_n=None):
        #Create axes.
        fig,ax = plt.subplots()
        ax.set_xlabel('Z (AU)')
        ax.set_ylabel('Rate')
        ax.set_yscale('log',nonposy='clip')

        #Handle colormap nonsense.
        if type(cmap_pro) == str:
            cmap_pro = get_cmap(cmap_pro)
        if type(cmap_des) == str:
            cmap_des = get_cmap(cmap_des)

        #Figure out how many reactions to load.
        if load_n is None:
            load_n = n

        #Rank rates. Take strongest n reactions.
        rates = self.rank_reacs(strmol,time,R)
        rates = rates[:load_n]

        #Count number of reactions producing and destroying strmol.
        n_pro = len(rates[:n][rates[:n][:,1] >= 0])
        n_des = len(rates[:n][rates[:n][:,1] <  0])

        #Load first reaction just to get z array.
        self.load_reac(strmol,rates[0,0],radii=R,times=time)
        z,_ = self.z_quant(rates[0,0],R=R,time=time)
        rt_pro = np.zeros_like(z)
        rt_des = np.zeros_like(z)

        pro = 0
        des = 0
        for rid,rate in rates:
            if rate >= 0:
                c = cmap_pro(1-pro/n_pro)
                pro += 1
            else:
                c = cmap_des(1-des/n_des)
                des += 1
            print("Loading %d: %s, %15.7E"%(int(rid),self.get_reac_str(rid),rate))
            self.load_reac(strmol,rid,times=time,radii=R)
            z,rt = self.z_quant(rid,R=R,time=time)
            if pro+des <= n:
                #Only plot n rates.
                ax.plot(z,rt,color=c,ls='--',label=self.get_reac_str(rid,fmt='latex'))
            if total:
                rt[np.isnan(rt)] = 0
                if rate >= 0:
                    rt_pro += rt
                else:
                    rt_des += rt
        if total:
            ax.plot(z,rt_des,color='red',label='Destruction Rate')
            ax.plot(z,rt_pro,color='blue',label='Prodution Rate')
        
        if not plot_mols is None:
            if type(plot_mols) == str:
                plot_mols = [plot_mols]
            sax = ax.twinx()
            sax.set_yscale('log',nonposy='clip')
            for mol in plot_mols:
                self.load_mol(mol,times=time)
                z,ab = self.z_quant(mol,R=R,time=time)
                sax.plot(z,ab,label=mol)

        ax.legend(loc=0)
