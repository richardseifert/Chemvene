import numpy as np
from time import time as timer
import glob

def load_rates(direc,strmol,reacs=None,times=None,radii=None,zones=None,min_rate = 0.):
    '''
    Function for loading reaction rate .rout files from the chemical code.
    
    ARGUMENTS:
        direc - String path to directory containing .rout files.
        strmol - Molecule name prefix on .rout files.
        reacs - Reaction IDs to load rates for.
        times - Values of time to consider.
        radii - NEED TO ADD. Values of radius to consider.
        min_rate - The minimum reaction rate to load.
        eventually:
            times -  t = 1-1E6 disk ages to load rates for.
    '''
    try:
        iter(reacs)
    except TypeError:
        if not reacs is None:
            reacs = [reacs]
    try:
        iter(times)
        times = np.array(times).astype(float)
    except TypeError:
        if not times is None:
            times = [times]
            times = np.array(times).astype(float)
    try:
        iter(radii)
        radii = np.array(radii).astype(float)
    except TypeError:
        if not radii is None:
            radii = [radii]
            radii = np.array(radii).astype(float)
    #Get list of rad
    print(direc+strmol+'_*.rout')
    fpaths = glob.glob(direc+strmol+"_*.rout")
    radnam = np.array([fpath.split("/")[-1].split("_")[-1].split('.rout')[0] for fpath in fpaths])
    radval = np.array([float(strg.rstrip()) for strg in radnam])
    #Sort by radius.
    sort = np.argsort(radval)
    radnam = radnam[sort]
    radval = radval[sort]
    fpaths = np.array(fpaths)[sort]

    if not radii is None:
        #Only load radii specified.
        nearest = np.argmin([ (radval - r)**2 for r in radii ], axis=1)
        nearest = np.unique(nearest) #Remove duplicate radii.
        radval = radval[nearest]
        fpaths = fpaths[nearest]


    dat = np.array([])
    for R,fpath in zip(radval,fpaths):
        a = load_rates_single(fpath,reacs,times,zones,min_rate)
        if len(a) == 0:
            continue
        a = np.insert(a,1,R,axis=1) #Insert column with current radius.
        try:
            dat = np.vstack([dat,a])
        except ValueError:
            dat = a.copy()
        
    return dat

def load_rates_single(fpath,reacs,times,zones=None,min_rate=0.):
    try:
        iter(zones)
        zones = np.array(zones).astype(int)
    except TypeError:
        if not zones is None:
            zones = [zones]
            zones = np.array(zones).astype(int)

    #Read rates file.
    f = open(fpath)
    lines = list(filter(None,f.read().split('\n')))
    f.close()

    expanded_lines = []
    keep = lambda v,vals,cast=lambda s:s : (vals is None) or (v in vals) or (cast(v) in vals)
    keep_time = lambda t: (times is None) or (np.abs(np.nanmin(np.log10(float(t))-np.log10(times))) < 1e-3)
    for line in lines:
        if line[0]=='#':
            continue #Ignore commented lines.
        info = list(filter(None, line.split(' ')))             
        zone = info[0]
        t = info[1]
        if not keep_time(t) or (not zones is None and not int(zone) in zones):
            continue
        for i in range(2,len(info),2):
            reac_id = info[i]
            reac_rate = info[i+1]
            if keep(reac_id,reacs,int) and abs(float(reac_rate)) > min_rate: 
                expanded_lines.append(' '.join([t,zone,reac_id,reac_rate])+'\n')
    if len(expanded_lines) == 0:
        return np.array([])
    a = np.loadtxt( (line for line in expanded_lines) )
    if len(a.shape) == 1:
        a = a[None,:]
    return a

def total_rates(direc,strmol,times=None,radii=None,zones=None,min_rate = 0.):
    '''
    Function for loading reaction rate .rout files from the chemical code.
    
    ARGUMENT:
        direc - String path to directory containing .rout files.
        strmol - Molecule name prefix on .rout files.
        times - Values of time to consider.
        radii - Values of radius to consider.
        min_rate - The minimum reaction rate to load.
    '''

    try:
        iter(times)
        times = np.array(times).astype(float)
    except TypeError:
        if not times is None:
            times = [times]
            times = np.array(times).astype(float)
    try:
        iter(radii)
        radii = np.array(radii).astype(float)
    except TypeError:
        if not radii is None:
            radii = [radii]
            radii = np.array(radii).astype(float)
    try:
        iter(zones)
        zones = np.array(zones).astype(float)
    except TypeError:
        if not zones is None:
            zones = [zones]
            zones = np.array(zones).astype(float)
            

    #Get list of radii
    fpaths = glob.glob(direc+strmol+"_*.rout")
    radnam = np.array([fpath.split("/")[-1].split("_")[-1].split('.rout')[0] for fpath in fpaths])
    radval = np.array([float(strg.rstrip()) for strg in radnam])
    #Sort by radius.
    sort = np.argsort(radval)
    radnam = radnam[sort]
    radval = radval[sort]
    fpaths = np.array(fpaths)[sort]

    if not radii is None:
        #Only load radii specified.
        nearest = np.argmin([ (radval - r)**2 for r in radii ], axis=1)
        radval = radval[nearest]
        fpaths = fpaths[nearest]

    rate_dict = {}
    keep = lambda v,vals,cast=lambda s:s : (vals is None) or (v in vals) or (cast(v) in vals)
    keep_time = lambda t: (times is None) or (np.abs(np.nanmin(np.log10(float(t))-np.log10(times))) < 1e-3)
    npath = 0
    for fpath in fpaths:
        npath+=1
        #print("%d / %d"%(npath,len(fpaths)))
        #Read rates file.
        f = open(fpath)
        lines = list(filter(None,f.read().split('\n')))
        f.close()
        for line in lines:
            if line[0]=='#':
                continue #Ignore commented lines.
            info = list(filter(None, line.split(' ')))
            zone = info[0]
            t = info[1]
            if not keep_time(t) or (not zones is None and not int(zone) in zones):
                continue
            for i in range(2,len(info),2):
                    reac_id = int(info[i])
                    reac_rate = float(info[i+1])
                    try:
                        rate_dict[reac_id] += reac_rate
                    except KeyError:
                        rate_dict[reac_id] = reac_rate

    
    rates = np.array(list(rate_dict.items()))
    rates = rates[np.argsort(-np.abs(rates[:,1]))]
    return rates

def get_reac_str(fpath,reac_id,fmt='ascii'):
    '''
    Function for obtaining a string representation of a chemical reaction from its
    reaction ID, given the reaction file.

    ARGUMENTS:
        fpath    - String path to the reaction file.
        reac_id  - int ID of the desired reaction.
        fmt      - Format of the reaction string created.
            -fmt = latex  :  Returns latex formatted string with
                             sub-/super-script numbers and charges.
            -fmt = ascii  :  Returns simple ascii formatted reaction string.
    RETURNS:
        reac_str - String representation of the reaction with the given ID.
    '''
    f = open(fpath)
    line = f.readline()
    while line[0] == '#':
        line = f.readline()
    found = False
    eof_count = 0
    while not found:
        try:
            info = line.split()
            id = int(info[0])
            found = id == reac_id
            line = f.readline()
        except:
            line = f.readline()
            if line == "":
                eof_count += 1
            else:
                eof_count = 0
            if eof_count >= 5:
                return "REACTION NOT FOUND"
    species = info[1:-4]
    if fmt == 'latex':
        # Return latex formatted string.
        sup = ['+']
        sub = [str(n) for n in range(10)]

        #Go through each species and add super-/sub-script characters.
        for i,sp in enumerate(species):
            new_sp = ''
            for c in sp:
                if c in sup:
                    new_sp += '$^{%s}$'%c
                elif c in sub:
                    new_sp += '$_{%s}$'%c
                else:
                    new_sp += c
            species[i] = new_sp

        #Join species into 2-reactant chemical formula, with latex arrow.
        strg = ' + '.join(species[:2])+' $\\rightarrow$ '+' + '.join(species[2:])
    else:
        # Return plain ascii string.
        strg = ' + '.join(species[:2])+' -> '+' + '.join(species[2:])
    return strg
                
if __name__ == '__main__':
    fpath = '../D_tests/ssm_r146.0626_D/HCO+_e1_146.0626.rout'
    fpath = '/bucket/ras8qnr/MasterChem_Phobos/runs/ssm/e1/rates/H3+_e1_372.4317.rout'
    fpath = '/bucket/ras8qnr/MasterChem_Phobos/runs/ssm/e1/rates/HCO+_e1_594.7036.rout'
    direc = '/Users/ras8qnr/Research/IM_Lup/mchem_snap/runs/ssm/e1/rates/'
    #rtmat = load_rates(fpath,min_rate=1e-17,reacs=8549,times=['4.230e+05'])
    #rtmat = load_rates(direc,"HCO+",min_rate=1e-17,times=3.34,reacs=5359)
    #rates = total_rates(direc,"HCO+",times=4.92e5)
    #rtmat = load_rates(fpath,min_rate=1e-17,reacs=8549)
    #np.savetxt("test.dat",rtmat)
