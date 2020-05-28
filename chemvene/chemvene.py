import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pandas import DataFrame
from scipy.interpolate import griddata
import glob
import os

#Package Imports
from .read.read_abunds import find_mol, load_mol_abund
from .read.read_rates import load_rates, get_reac_str, total_rates
from .read.read_radfields import load_radfield
from .read.read_lambda import read_levels,read_trans
from .misc import contour_points, get_contour_arr, remove_nan, sigfig, iterable, nint
from chemvene import __path__ as pkg_path

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
            outdir  - string path to the runs/ directory where model output is
                      stored.
            environ - string path to the environ/ directory used to run your
                      chemical model. (Must given it outdir/environ doesn't exist)
            inp     - string filename of the input file used to run your model.
                      (Must be given if outdir/0io* doesn't exits)
    '''

    ################################################################################
    ################################ Initialization ################################
    ################################################################################

    def __init__(self,outdir,environ=None,inp=None,bsd=bsd):
        self.outdir = outdir
        if self.outdir[-1] != '/':
            self.outdir += '/'

        if not environ is None:
            self.set_environ(environ)
        elif os.path.exists(self.outdir+'environ/'):
            self.set_environ(self.outdir+'environ/')
        else:
            raise FileNotFoundError("Could not determine environ/ directory to use for this model.")

        if not inp is None:
            self.set_environ(environ)
        else:
            outdir_0io_paths = glob.glob(self.outdir+'0io*')
            if len(outdir_0io_paths) > 0:
                self.set_inp(outdir_0io_paths[0].split('/')[-1])
            else:
                raise FileNotFoundError("Could not determine 0io file to use for this model.")
        self.phys = DataFrame()
        self.radfields = {}
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

    def copy(self):
        '''
        Make a hard copy of a chem_mod instance.
        '''
        #Initialize
        new_inst = chem_mod(outdir=self.outdir,environ=self.environ,inp=self.inp)
        
        #Hard copy physical quantities
        new_inst.phys = self.phys.copy()
        #for q in self.phys.columns:
        #    new_inst.set_quant(q,self.phys[q])

        #Hard copy abundances
        for mol in self.abunds.keys():
            new_inst.abunds[mol] = self.abunds[mol].copy()

        #Hard copy rates
        for rid in self.rates.keys():
            new_inst.rates[rid] = self.rates[rid].copy()

        return new_inst

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
            #Match by nearest R and zAU.
            #  pandas.DataFrame.merge has failed me in this reguard.. :(
            #  So I just had to do it myself, huney, using griddata.
            merge_cols = [col for col in tbl.columns if not col in self.phys.columns]
            points = np.vstack([tbl['R'],tbl['zAU']]).T
            values = np.array(tbl[merge_cols])
            phys_points = np.array([self.phys['R'],self.phys['zAU']]).T
            matched = griddata(points,values,phys_points,method='nearest')
            merged = self.phys.copy()
            for i,col in enumerate(merge_cols):
                merged[col] = matched[:,i]
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
        print(env_paths[0])

        #Determine number of shells in model.
        f1 = open(env_paths[0])
        for i,line in enumerate(f1):
            if i==2:
                nshells = int(line.strip())
                f1.close()
                break

        dat = np.array([])
        shells = np.array([np.arange(nshells)+1]).T
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

    def load_field(self,field,path=None):
        if path is None:
            path = self.inp_paths[field]
        print("Loading %s field from: %s"%(field,path))
        dat = load_radfield(path)
        R    = dat[:,0]
        zAU  = dat[:,1]
        spec = dat[:,2]
        flux = dat[:,3]    

        self.radfields[field] = DataFrame()
        spec_vals = np.unique(spec)
        for sv in spec_vals:
            mask = spec==sv
            tbl = DataFrame()
            tbl['R']    = R[mask]
            tbl['zAU']  = zAU[mask]
            tbl['flux'] = flux[mask]
            self.radfields[field][sv] = self.merge(tbl)['flux']

    ################################################################################
    ####################### Handling Abundances of Species #########################
    ################################################################################

    def limedir(self,strmol):
        '''
        Function that produces string limefg path for a given species.
        It's a pretty pointless method, because I only need the limefg path
        twice, when loading and writing species abundances. But, I figured
        if I ever want to change where I save limefg or what I want to rename
        the directory, I can just change it once in this method.

        ARGUMENTS:
            strmol - String name of the species.

        RETURNS:
            string path of a directory where limefg should go.
        '''
        return self.outdir+'e1/limefg_'+strmol+'/'

    def grab_mol(self,strmol,*args,**kwargs):
        if not strmol in self.abunds:
            self.load_mol(strmol,*args,**kwargs)

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
        if times is None:
            times = all_times
        else:
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
            self.abunds[strmol][time] = merged['abund']/2
            # ^ factor of 2 because LIME wants abundance per H2 instead of per H
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

    def write_mol(self,strmol,label=None):
        '''
        Method that writes abundances for a species in the limefg format
        used by LIME radiative transfer.

        ARGUMENTS:
            strmol - string name of the species to load.
        '''
        if not strmol in self.abunds.keys():
            self.read_mol(strmol)
        if label is None:
            label = strmol
        else:
            label = strmol+'_'+label
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

        
        limedir = self.limedir(label)
        if not os.path.exists(limedir):
            os.makedirs(limedir)
        times = np.sort(np.unique(self.abunds[strmol].columns))
        for i,time in enumerate(times):
            fname=limedir+strmol+'_time'+str(i)+'.dat'
            abu = 2*np.array(self.abunds[strmol][time]) 
            # ^ factor of 2 because LIME wants abundance per H2, not per H.
            abu[(savetbl['rho'] <= 1e2) | (abu < 1e-32)] = 0.0
            savetbl.loc[:,'abund'] = abu
            no_nan = remove_nan(self.phys['R'],abu)
            savearr = np.array(savetbl)[no_nan]
            
            np.savetxt(fname,savearr,fmt='%15.7E')

    ################################################################################
    ######################### Handling Chemical Reactions ##########################
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

    def load_reac(self,strmol,reacs,times=None,radii=None,zones=None):
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
        dat = load_rates(self.outdir+'e1/rates/',strmol,reacs,times,radii,zones)

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

    def rank_reacs(self,strmol,time=None,R=None,zone=None):
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
        rates = total_rates(self.outdir+'e1/rates/',strmol,times=time,radii=R,zones=zone)
        return rates

    ################################################################################
    ############################# Requesting Model Data ############################
    ################################################################################

    def get_quant(self,quant,time=0,mask=None,fmt='pandas'):
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
        elif quant in self.rates.keys():
            times = np.array(self.rates[quant].columns)
            #nearest = self.nearest_times(times)    #Times won't necessarily align with rates columns. Figure that out first.
            nearest = times[np.argmin((times-time)**2)]
            quant = self.rates[quant][nearest]
            if np.nanmean(quant) < 0:
                quant = -quant
        elif quant[0]=='n' and quant[1:] in self.abunds.keys():
            quant = self.get_mol_dens(quant[1:],time)
        else:
            raise ValueError("The quantity %s was not found for this model."%(quant))

        if mask is None:
            mask = np.ones_like(quant).astype(bool)
        elif fmt == 'contour':
            raise ValueError("Cannot return contour-formatted arrays with mask")

        if fmt == 'pandas':
            return quant[mask]
        elif fmt == 'contour':
            nx = len(list(set(self.phys['R'])))
            ny = len(list(set(self.phys['shell'])))
            return get_contour_arr(quant,nx,ny,sortx=self.phys['R']) 
        else:
            raise ValueError("Unrecognized format: %s"%(fmt))

    def get_spatial(self,yaxis='z',fmt='pandas'):
        R = self.get_quant('R',fmt=fmt)

        Y = self.get_quant('zAU',fmt=fmt)
        if yaxis == 'z/r':
            Y = Y/R
        elif yaxis == 'zone':
            Y = 50. - (Y/R)/np.nanmax(Y)*49.
        elif not yaxis=='z':
            raise ValueError("Unrecognized yaxis: %s"%(yaxis))
        return R,Y

    def get_sigma(self):
        R = self.phys['R']
        zAU = self.phys['zAU']
        rho = self.phys['rho']
        
        cumulative = {}
        for r in R:
            if not r in cumulative.keys():
                cumulative[r] = 0.0

        sort = np.argsort(zAU)
        for i in range(len(R)):
            pass
    
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
    
    def R_quant(self,quant,zone=0,time=0):
        '''
        Method for obtaining quant as a function of radius at a particular zone and time.

        ARGUMENTS:
            quant - The quantity you're interested in. Could be physical quantity,
                    chemical species for abundances, or reaction ID for rates.
            zone - Shell at which to return quant. Default is zone = 0, the
                    outer layer of the disk.
            time  - Time at which to return quant. Defaults to first timestep.
        Returns
            R     - 1D radii in AU.
            quant - 1D quant values corresponding to R.
        '''
        #Copy quant string before it's overwritten.
        quant_str = (str(quant)+'.')[:-1]  #This weirdness is to force python to hardcopy the string.

        #Find nearest R value in grid.
        zones = np.array(list(set(self.phys['shell'])))
        zone = zones[np.argmin(np.abs(zones-zone))]

        #Get 1-D arrays of z and quant at specified R value.
        zone_mask = self.phys['shell'] == zone
        R = np.array(self.phys['R'][ zone_mask ])
        quant = self.get_quant(quant,time)
        quant = np.array(quant[ zone_mask ])

        #Sort by z
        sort = np.argsort(R)
        R = R[sort]
        quant = quant[sort]

        return R,quant

    def get_mol_dens(self, strmol, time=0):
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
        try:
            nX = self.get_quant('n'+strmol,time=time)
        except:
            nX = self.get_quant(strmol,time=time)
            

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

    def optical_depth(self,strmol,trans,lambdafile=None,time=0):
        '''
        Method for producing columnd density profile for a given species.

        ARGUMENTS:
            strmol - string of the molecule you want to get column density of.
            time   - timestep you want columnd density at.
        RETURNS:
            R_vals - Radius values.
            cd     - Corresponding column densities at those radii.
        '''
        #Define some relevant constants
        h = 6.6260755e-27 #erg s
        c = 2.99792458e10 #cm s^-1
        kb = 1.380658e-16 #erg K^-1

        #Load number density of strmol (cm^-3).
        try:
            nX = self.get_quant('n'+strmol,time=time)
        except:
            nX = self.get_quant(strmol,time=time)
            
        #Load corresponding disk locations.
        zone = np.array(self.get_quant('shell'))
        zone_vals = np.unique(zone)
        print(zone_vals)
        zone_vals = zone_vals[np.argsort(zone_vals)]
        print(zone_vals)
        Rarr = np.array(self.get_quant('R'))
        Zarr = np.array(self.get_quant('zAU'))
        # and temperatures!
        Tarr = np.array(self.get_quant('Tgas'))

        #Read lambda file
        levels = read_levels(lambdafile)
        transitions = read_trans(lambdafile)
        #  Get list of energies and statistical weights for each level.
        En = levels[:,1]*h*c
        gn = levels[:,2]
        #  Get constants for this transition.
        A = transitions[trans,3] #Aul for this transition.
        print('Einstein A (s^-1)',A)
        freq = transitions[trans,4]*1e9
        print('Frequency:',freq/1e9,'GHz')
        lam = c/freq
        print('Wavelength:',lam,'cm')
        #   get upper- and lower-state energies and statistical weights
        Elower=En[0]
        glower=gn[0]
        Eupper=En[1]
        gupper=gn[1]
        for lid,E,g in zip(levels[:,0],En,gn):
            if lid==transitions[trans,1]:
                Eupper = E
                gupper = g
            if lid==transitions[trans,2]:
                Elower = E
                glower = g
        grat = gupper/glower

        #Functions so evaluate at each location.
        partition_func = lambda T,E=En: np.array([np.sum(np.exp(-E/(kb*temp))) for temp in T])
        integrand = lambda n,T: n/partition_func(T)*glower*np.exp(-Elower/(kb*T))*(1-np.exp(-h*freq/(kb*T)))

        tau = np.zeros_like(zone)
        for i,zn1,zn2 in zip(np.arange(len(zone_vals)-1),zone_vals[:-1],zone_vals[1:]):
            in_zn1 = zone == zn1
            in_zn2 = zone == zn2
            n1 = nX[in_zn1]
            n2 = nX[in_zn2]
            Z1 = Zarr[in_zn1]
            Z2 = Zarr[in_zn2]
            T1 = Tarr[in_zn1]
            T2 = Tarr[in_zn2]
            tau[in_zn2] = tau[in_zn1] + grat*A/(8*np.pi) * lam**2 * 0.5*(Z1-Z2)*(integrand(n1,T1)+integrand(n2,T2))

        return tau
            
    def get_spec(self,field,r,z):
        R = self.get_quant('R')
        R_vals = np.unique(R)
        nearest_R = R_vals[np.argmin(np.abs(R_vals-r))]
        Z = self.get_quant('zAU')
        Z_vals = Z_vals[R == nearest_R]
        nearest_Z = Z_vals[np.argmin(np.abs(Z_vals-z))]
        mask = (R == nearest_R) & (Z == nearest_Z)
        
        spec_all = self.radfields[field]
        spec_vals = np.sort(spec_all.columns)
        intensity_vals = np.zeros_like(spec_vals)
        for i,sv in enumerate(spec_vals):
            intensity_vals[i] = spec_all[sv][mask]
        return spec_vals,intensity_vals

    ################################################################################
    ############################# Altering Model Data ##############################
    ################################################################################

    def set_quant(self,quant,val,mask=None,time=None):
        '''
        Method for setting a model quantity (e.g. Tgas, CO abudance, etc.) to a new
        value or set of values, optionally within a masked region only.

        ARGUMENTS:
            quant - The quantity to be changed
                    Must be found in either self.phys, self.abunds, or self.rates
            val   - The new value or array of values for this quantity
            mask  - A pre-generated mask for where the quantity should be changed.
                    Default, no masking.
              Ex.) #Enhancing model CO abundance within 50 AU
                   cmod = chem_mod(someoutdir)
                   cmod.grab_mol('CO')
                   R = cmod.get_quant('R')
                   mask = R < 50    # Returns array of Trues and Falses.
                   cmod.set_quant('CO',1e-4,mask=mask)
        '''
        #If mask not given, make an all-True mask (equiv to not masking).
        if type(mask) is type(None):
            mask = np.ones_like(self.phys['R']).astype(bool)

        #If val is another chem_mod instance, take the masked quant from
        # that chem_mod instance
        if isinstance(val,chem_mod):
            val = val.get_quant(quant,time=time if not time is None else 0)

        if quant in self.phys.columns:
            self.phys[quant][mask] = val
        elif quant in self.abunds.keys():
            times = np.array(self.abunds[quant].columns)
            if time is None:
                for t in times:
                    self.abunds[quant][t][mask] = val
            else:
                nearest = times[np.argmin((times-time)**2)]
                self.abunds[quant][nearest][mask] = val
        elif quant in self.rates.keys():
            times = np.array(self.rates[quant].columns)
            if time is None:
                for t in times:
                    self.rates[quant][t][mask] = val
            else:
                nearest = times[np.argmin((times-time)**2)]
                self.rates[quant][nearest][mask] = val
        elif quant[0]=='n' and quant[1:] in self.abunds.keys():
            #Compute abundances corresponding to given density, val.
            rho = self.get_quant('rho')
            nH = rho/mp
            ab_val = val/nH

            times = np.array(self.abunds[quant[1:]].columns)

            if time is None:
                for t in times:
                    self.abunds[quant[1:]][t][mask] = ab_val
            else:
                nearest = times[np.argmin((times-time)**2)]
                self.abunds[quant[1:]][nearest][mask] = ab_val
            
        else:
            raise ValueError("The quantity %s was not found for this model."%(quant))

    def set_all(self,other,mask):
        '''
        Replace model quantities with those of another chem_mod instance within a
        specified mask.

        ARGUMENTS:
            other - A chem_mod instance.
            mask  - An array of Trues and Falses to be used when setting model
                    quantities to those of the given cmod, other.
        '''
        #Set physical quantities
        for q in self.phys.columns:
            try:
                self.set_quant(q,other.get_quant(q)[mask],mask=mask)
            except ValueError:
                print("Warning: Quantity %s was not found for this model and is not \
                       being set."%(q))

        for mol in self.abunds.keys():
            times = np.array(self.abunds[mol].columns)
            for t in times:
                try:
                    self.set_quant(mol,other.get_quant(mol,time=t)[mask],mask=mask,time=t)
                except ValueError:
                    print("Warning: Quantity %s at time %s was not found for this model and is not \
                           being set."%(mol,t))
        for rid in self.rates.keys():
            times = np.array(self.rates[rid].columns)
            for t in times:
                try:
                    self.set_quant(rid,other.get_quant(rid,time=t)[mask],mask=mask,time=t)
                except ValueError:
                    print("Warning: Quantity %s at time %s was not found for this model and is not \
                           being set."%(rid,t))
            
    ################################################################################
    ################################## Plotting ####################################
    ################################################################################

    def profile_quant(self,quant,time=0,vmin=None,vmax=None,plot_grid=False,yaxis='z',xscale='linear',yscale='linear',return_artist=False,**kwargs):
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
        R,Y = self.get_spatial(yaxis=yaxis)
        if yaxis == 'z/r':
            ylabel = 'Z/R'
        elif yaxis == 'zone':
            ylabel = 'Zone'
        else:
            ylabel = 'Z (AU)'
        if vmin is None:
            vmin = np.nanmin(quant[quant>0])
        if vmax is None:
            vmax = np.nanmax(quant[quant>0])
        nx = len(list(set(self.phys['R'])))
        ny = len(list(set(self.phys['shell'])))

        if return_artist:
            ax,cont = contour_points(R,Y,quant,nx=nx,ny=ny,vmin=vmin,vmax=vmax,return_artist=True,**kwargs)
        else:
            ax = contour_points(R,Y,quant,nx=nx,ny=ny,vmin=vmin,vmax=vmax,**kwargs)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if plot_grid:
            ax.scatter(R,Y,s=1,color='black')
        ax.set_xlabel('R (AU)')
        ax.set_ylabel(ylabel)

        if return_artist:
            return ax,cont
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

    def plot_best_reacs(self,strmol,n,R=None,zone=None,time=None,plot_mols=None,\
                        total=True,cmap_pro='Blues',cmap_des='Reds',load_n=None,\
                        ls_pro='--',ls_des='-.',ax=None):

        if not R is None and not zone is None:
            raise ValueError("Both R and zone cannot be given; give one or the other.")
        #Create axes.
        if ax is None:
            fig,ax = plt.subplots()
        ax.set_ylabel('Rate')
        ax.set_yscale('log',nonposy='clip')
        if not R is None:
            ax.set_xlabel('Z (AU)')
        if not zone is None:
            ax.set_xlabel('R (AU)')

        #Handle colormap nonsense.
        if type(cmap_pro) == str:
            cmap_pro = get_cmap(cmap_pro)
        if type(cmap_des) == str:
            cmap_des = get_cmap(cmap_des)

        #Figure out how many reactions to load.
        n = int(n)
        if load_n is None:
            load_n = n

        #Rank rates. Take strongest n reactions.
        rates = self.rank_reacs(strmol,time,R=R,zone=zone)
        rates = rates[:load_n]

        #Count number of reactions producing and destroying strmol.
        n_pro = len(rates[:n][rates[:n][:,1] >= 0])
        n_des = len(rates[:n][rates[:n][:,1] <  0])

        pro = 0
        des = 0
        for rid,rate in rates:
            if rate >= 0:
                c = cmap_pro(1-pro/n_pro)
                ls = ls_pro
                pro += 1
            else:
                c = cmap_des(1-des/n_des)
                ls = ls_des
                des += 1
            print("Loading %d: %s, %15.7E"%(int(rid),self.get_reac_str(rid),rate))
            #try:
            self.load_reac(strmol,rid,times=time,radii=R,zones=zone)
            #except IndexError:
            #    print ("Warning: Couldn't load %s"%(rid))
            #    continue
            if not R is None:
                x,rt = self.z_quant(rid,R=R,time=time)
            if not zone is None:
                x,rt = self.R_quant(rid,zone=zone,time=time)
            if pro+des <= n:
                #Only plot n rates.
                ax.plot(x,rt,color=c,ls=ls,label="%d: %s"%(rid,self.get_reac_str(rid,fmt='latex')))
            if total:
                try:
                    rt_pro
                    rt_des
                except NameError:
                    rt_pro = np.zeros_like(x)
                    rt_des = np.zeros_like(x)
                rt[np.isnan(rt)] = 0
                if rate >= 0:
                    rt_pro += rt
                else:
                    rt_des += rt
        if total:
            ax.plot(x,rt_des,color='red',label='Destruction Rate')
            ax.plot(x,rt_pro,color='blue',label='Prodution Rate')
        
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

        return ax
