import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
import glob
import os

#Package Imports
from .read.read_abunds import find_mol, load_mol_abund
from .read.read_rates import load_rates, get_reac_str, total_rates
from .misc import contour_points, remove_nan, sigfig, iterable
from chem_mod import __path__ as pkg_path

#Path to the Chemical Code Directory.
bsd = '/bucket/ras8qnr/MasterChem_Phobos/'

#Some constants that get used throughout.
mp = 1.67e-24  #Mass of proton in g
mau = 1.496e11 #Conversion from AU to meters.

class chem_mod:
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
        print(self.inp)
        self.inp_paths = {k:None for k in ['spec','reac','uv','xray','isrf','rn']}
        d = np.genfromtxt(self.inp,dtype=str)
        for i,k in enumerate(self.inp_paths.keys()):
            if os.path.exists(bsd+d[i]):
                self.inp_paths[k] = bsd+d[i]
    def load_physical(self):
        '''
        Load the input physical model from 1environ files
        located in the environ directory.
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
    def load_times(self):
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

    def merge(self,tbl):
        '''
        Prepare a given table to be merged according to position, R and zAU.

            ARGUMENTS:
              tbl - A pandas table containing the two columns 'R' and either 'shell'.

            RETURNS:
              A tbl with the same number of rows as phys. The returned table
              has values ordered according to phys['R'] and phys['shell']
        '''

        #Match R values to their nearest R values in phys['R'].
        #This is necessary to merge tables properly.
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
        #Set times to the exact model times.
        # (For some reason, they vary slightly in the r*.out and *.rout files).
        ctimes = tbl.columns
        mtimes = self.nearest_times(ctimes,itr=True)
        return tbl.rename(columns=dict(zip(ctimes,mtimes)))
    def load_phot(self,uv=True,xray=True):
        #I don't quite remember what this was going to be.. :(
        #Something like load the uv and xray profiles, but it gets
        #weird because they're spectra as well, so there's an extra axis.
        pass
    def load_mol(self,strmol,times=None):
        limedir = self.limedir(strmol)
        if not os.path.exists(limedir):
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
        if not strmol in self.abunds.keys():
            self.read_mol(strmol)
        savetbl = self.phys[['R','zAU','rho','Tgas','Tdust']]
        savetbl.loc[:,'rho'] *= 0.8/(2.0*mp) * 1e6
        savetbl.loc[:,'abund'] = np.zeros_like(savetbl['R']) #Place holder.

        # Match tmp table and physical table by positions.
        #tmp = np.genfromtxt(bsd+"scripts/IMLupModelV8_gaia/imlup_gaia_v2_abrig_model_Tgas_SB_G04.txt")
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
            #savetbl.loc[:,'abund'] = self.abunds[strmol][time]
            #savetbl.loc[(savetbl['rho'] <= 1e4) | (savetbl['abund'] < 1e-28), 'abund'] = 0.0
            abu = np.array(self.abunds[strmol][time])
            #print("Time = %f, Before filter: %15.7E"%(time,np.nanmean(abu)))
            abu[(savetbl['rho'] <= 1e4) | (abu < 1e-28)] = 0.0
            #abu[(savetbl['abund'] < 1e-28)] = 0.0
            #print("Time = %f, After filter: %15.7E"%(time,np.nanmean(abu)))
            savetbl.loc[:,'abund'] = abu
            no_nan = remove_nan(self.phys['R'],self.phys['shell'],abu)
            savearr = np.array(savetbl)[no_nan]
            np.savetxt(fname,savearr,fmt='%15.7E')
    def limedir(self,strmol):
        return self.outdir+'e1/limefg_'+strmol+'/'
    def load_reac(self,strmol,reacs,times=None,radii=None):
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
    def get_reac_str(self,reac_id,fmt='ascii'):
        return get_reac_str(self.inp_paths['reac'], reac_id, fmt)
    def rank_reacs(self,strmol,time=None,R=None):
        if not time is None:
            time = self.nearest_times(time)
        rates = total_rates(self.outdir+'e1/rates/',strmol,time,R)
        return rates
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

    def shell_quant(self,quant,shell,ax=None):
        R = self.phys['R']
        quant = self.get_quant(quant)
        mask = self.phys['shell'] == shell
        R,quant = R[mask],quant[mask]
        sort = np.argsort(R)
        R,quant = R[sort],quant[sort]
        return R,quant

    def get_quant(self,quant,time=0):
        if iterable(quant) and type(quant) != str:
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
        else:
            raise ValueError("The quantity %s was not found for this model."%(quant))
        return quant
    def profile_quant(self,quant,time=0,log=True,ax=None,vmin=None,vmax=None,levels=25,cmap='jet',plot_grid=False):
        quant = self.get_quant(quant,time)
        R = self.phys['R']
        z = self.phys['zAU']
        if vmin is None:
            vmin = np.nanmin(quant[quant>0])
        if vmax is None:
            vmax = np.nanmax(quant[quant>0])
        nx = len(list(set(self.phys['R'])))
        ny = len(list(set(self.phys['shell'])))
        #ax = contour_points(R,z,quant,nx=nx,ny=ny,ax=ax,vmin=vmin,vmax=vmax,levels=levels,cmap=cmap,locator=LogLocator())
        ax = contour_points(R,z,quant,nx=nx,ny=ny,ax=ax,log=log,vmin=vmin,vmax=vmax,levels=levels,cmap=cmap) 
        if plot_grid:
            ax.scatter(R,z,s=1,color='black')
        ax.set_xlabel('R (AU)')
        ax.set_ylabel('Z (AU)')
        return ax
    def profile_reac(self,reac,time=0,**kwargs):
        ax = self.profile_quant(reac,time=time,**kwargs)
        ax.set_title( self.get_reac_str(reac,fmt='latex') )
    def z_quant(self,quant,R=100,time=0):
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
