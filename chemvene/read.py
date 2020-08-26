import numpy as np
import glob

################################################################################
############################# Read Abundances ##################################
################################################################################
def find_mol(fpaths, strmol):
    '''
    Locate the abundances of the requested molecule, strmol. Return start row, 
    number of rows, column index, and associated time steps. 
    
    Broken radii may present incorrect times and abundances, so search each file 
    until a valid file is found.
    '''
    strmol = ensure_pad(strmol) #Pad strmol for search
    fi = 0
    bad_times = True
    #For some files, all the times are 0 and abundances are nan.
    #Go through files until there's one that's not like that.
    while bad_times and fi < len(fpaths):
        mol_head,nrows,col,times = find_mol_helper(fpaths[fi],strmol)
        bad_times = np.all(times.astype(float) == 0)
        fi += 1
    return mol_head,nrows,col,times
def find_mol_helper(fpath,strmol):
    '''
    For a given abundance file, locate the abundances of the requested molecule,
    strmol. Return start row, number of rows, column index, and associated time
    steps.
    '''
    #Read lines from file.
    f = open(fpath)
    lines = f.read().split('\n')
    f.close()

    #Make sure strmol is padded with spaces.
    strmol = ensure_pad(strmol)

    #Find all lines containing strmol.
    lnums = []
    for i,line in enumerate(lines):
        if strmol in line:
            lnums.append(i)
    mol_head = lnums[1]+1       #The +1 is to get rid of header.
    nrows = lnums[2]-lnums[1]-2 #The -2 is to get rid of header and footer.
    
    #Load the chunk of the file with strmol data in it.
    dat = np.genfromtxt(fpath, comments='--',skip_header=mol_head-1,max_rows=nrows+1,dtype=str)

    #Extract the column with strmol.
    header = list(dat[0,:])
    col = header.index(strmol.rstrip().lstrip())#np.where(header == strmol)
 
    #Cut off header line and get times.
    dat = dat[1:,:]
    times = dat[:,0]

    return mol_head,nrows,col,times
def ensure_pad(strg):
    return ' '+strg.strip()+' '

def load_mol(fpath,strmol,mol_head=None,nrows=None,col=None):
    '''
    Load abundances for strmol from the abundance file fpath.
    '''
    if mol_head is None or nrows is None or col is None:
        mol_head,nrows,col,_ = find_mol(fpath,strmol)
    dat = np.genfromtxt(fpath, comments='--',skip_header=mol_head,max_rows=nrows,dtype=str)
    abund = dat[:,col].astype(float)
    abund[np.isnan(abund)] = 0.
    return abund

def load_mol_abund(direc,strmol):
    #Get paths to r.out files.
    if direc[-1] != '/':
        direc += '/'
    fpaths = glob.glob(direc+'r*.out')
    
    #Get location of strmol in r.out files, and timesteps.
    mol_start,nrows,col,Times = find_mol(fpaths,strmol)
    nTimes = len(Times)

    #Get list of radii.
    fpaths = glob.glob(direc+"r*_1.out")
    radnam = np.array([fpath.split("/")[-1].split("_e1")[0][1:] for fpath in fpaths])
    radval = np.array([float(strg.rstrip()) for strg in radnam])
    #Sort by radius.
    sort = np.argsort(radval)
    radnam = radnam[sort]
    radval = radval[sort]
    nR = len(radnam)

    # Find minimum z value from filenames of .out files.
    fpaths = glob.glob(direc+"r"+radnam[0]+"*.out")
    zval = [int(fpath.split("/")[-1].split("_")[-1].split(".")[0]) for fpath in fpaths]
    min_z = min(zval)
    #     Check all radii for the number of shells. Sometimes bad radii have too few.
    nZ = max([len(glob.glob(direc+"r"+rn+"*.out")) for rn in radnam])

    #Find where molecule abundance table starts. Don't overwrite Times,
    # because sometimes they are all zeros for some reason and that
    # breaks everything. "Not sure why" -Richard 06-11-19
    mol_start,nrows,col,_ = find_mol(fpaths,strmol)
    nTimes = len(Times)

    #Find all abundance files.
    fpaths = glob.glob(direc+'r*.out')
    #From each file, 4-D datapoints (X,y) will be extracted.
    #
    #   X = (t,R,z), t - Time in yr
    #                R - Radius in AU
    #                z - Height zone, unitless. min z is the disk surface
    #                                           max z is the disk midplane
    #   y = ab       ab- Abundance in number per H atom.
    #
    #Create array to store points.
    dat = np.zeros((len(fpaths)*nTimes,4))
    row_i = 0
    for i,path in enumerate(fpaths):
        row_f = row_i+nTimes
        dat[row_i:row_f,0] = Times
        try:
            rau,zau,Tg,Td,rho = load_physical(path) 
        except TypeError:
            print("Warning: The file at %s does not look like chemical model output to me :("%(path))
            row_i = row_f
            continue
        z = int(path.split('/')[-1].split('_')[-1].split('.')[0])
        ab = load_mol(path,strmol,mol_start,nrows,col)
        dat[row_i:row_f,1] = rau
        dat[row_i:row_f,2] = z
        if z == min_z:
            dat[row_i:row_f,3] = 0. #Assert surface abundance is 0.
        else:
            dat[row_i:row_f,3] = ab
        row_i = row_f
    return dat
def load_physical(fpath):
    '''
    For a given abundance file, read physical conditions from file header.
    '''
    f = open(fpath)
    line = f.readline()
    eof_count = 0
    while not "INITIAL VALUES:" in line:
        line = f.readline()
        if line == '':
            eof_count += 1
        else:
            eof_count = 0
        if eof_count > 5:
            raise TypeError("The file at %s does not look like chemical model output to me :("%(fpath))
    f.readline()
    f.readline()
    rau = np.float(f.readline().split('=')[1].split()[0])
    height = np.float(f.readline().split('=')[1].split()[0])
    f.readline()
    Tg = np.float(f.readline().split('=')[1].split()[0][:-1])
    Td = np.float(f.readline().split('=')[1].split()[0][:-1])
    rho = np.float(f.readline().split('=')[1].split()[0])
    f.close()
    return rau,height,Tg,Td,rho


################################################################################
########################## Read Radiation Fields ###############################
################################################################################
def load_radfield(path):
    '''
    Function for loading a radiation field output by the Bethell code.

    ARGUMENTS:
      path - Path to file containing radiation field.
    
    RETURNS:
      dat  - np.ndarray containing loaded radiation field.
             Four Columns:
               1 - Radius(AU)
               2 - Height(AU)
               3 - Wavelength/Energy
               4 - Flux per unit Wavelength/Energy
    '''
    #Read once to get wavelength/energy domain
    # and to check that file has expected format..
    f = open(path,'r')
    f.readline() #First line is garbage
    line = f.readline()
    assert (line.split(' ')[0]=='Radius(AU)'),'Expected radius line, got %s'%(line)
    line = f.readline()
    assert (line.split(' ')[0]=='z(AU)'),'Expected height z line, got %s'%(line)
    line = f.readline()
    specq = line.split(' ')[0]
    is_spectral = specq in ['Wavelength','Energy(keV)']
    assert is_spectral,'Expected spectral unit line, got %s'%(line)
    #Generate list of spec (Wavelength or Energy) values.
    spec_vals = []
    line = f.readline()
    while line.split(' ')[0]!='Radius(AU)' and len(line)>0:
        specv = list(filter(None,line.split(' ')))[0]
        try:
            spec_vals.append(float(specv))
        except ValueError:
            raise ValueError('Expected float %s value, got %s'%(specq,specv))
        line = f.readline()
    
    #Rewind to start of file.
    f.seek(0)

    #Read again, and collect spectrum for each position in the disk.
    dat = np.empty((4,0))
    f.readline() #First line is garbage
    line=f.readline()
    while not len(line)==0:
        #Read chunk header and get R and Z locations.
        info = list(filter(None,line.split(' ')))
        assert (info[0]=='Radius(AU)'),'Expected radius line, got %s'%(line)
        R = float(info[1])
        line=f.readline()
        info = list(filter(None,line.split(' ')))
        assert (info[0]=='z(AU)'),'Expected height z line, got %s'%(line)
        z_arr = np.array(info[1:]).astype(float)
        R_arr = R*np.ones_like(z_arr) 
        f.readline() #Ignore spec/flux unit line.
        line = f.readline()
        while line.split(' ')[0]!='Radius(AU)' and len(line)>0:
            #Read chunk data and get wavelengths and fluxes.
            info = list(filter(None,line.split(' ')))
            specv = float(info[0])
            flux_arr = np.array(info[1:]).astype(float)
            spec_arr = specv*np.ones_like(flux_arr)

            arr = np.array([R_arr,z_arr,spec_arr,flux_arr])
            dat = np.append(dat,arr,axis=1)

            line = f.readline()
    return dat.T

################################################################################
########################### Read Reaction Rates ################################
################################################################################
def load_rates(direc,strmol,reacs=None,times=None,radii=None,zones=None,min_rate = 0.):
    '''
    Function for loading reaction rates from a given directory for all reactions involving
     a molecule strmol.

    Loading can be drastically sped up by targeting specific radii, and moderately sped up 
     by targeting specific zones, times, or reaction IDs.
    
    ARGUMENTS:
        direc - String path to directory containing .rout files.
        strmol - Molecule name prefix on .rout files.
        reacs - Reaction IDs to load rates for. Default will load all rates found.
        times - Time steps to load rates for. Default all.
        radii - Radii to load rates for. 
              Rates are stored in files by radius, so specifiying radii drastically 
              speeds load time by limiting number of files to be opened.
        zones - Zones to load rates for.
        min_rate - The minimum reaction rate to load.
    '''

    #Ensure input datatypes!
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

    #Get list of radii
    fpaths = glob.glob(direc+strmol+"_*.rout")
    radnam = np.array([fpath.split("/")[-1].split("_")[-1].split('.rout')[0] for fpath in fpaths])
    radval = np.array([float(strg.rstrip()) for strg in radnam])
    #Sort by radius.
    sort = np.argsort(radval)
    radnam = radnam[sort]
    radval = radval[sort]
    fpaths = np.array(fpaths)[sort]

    #Only use requested radii.
    if not radii is None:
        #Only load radii specified.
        nearest = np.argmin([ (radval - r)**2 for r in radii ], axis=1)
        nearest = np.unique(nearest) #Remove duplicate radii.
        radval = radval[nearest]
        fpaths = fpaths[nearest]

    #Load rates!
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
def load_rates_single(fpath,reacs=None,times=None,zones=None,min_rate=0.):
    '''
    Load reaction rates for a single-radius .rout file!

    ARGUMENTS:
        reacs - Reaction IDs to load rates for. Default will load all rates found.
        times - Time steps to load rates for. Default all.
        zones - Zones to load rates for.
        min_rate - The minimum reaction rate to load.
    '''

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
    Function for quickly summing reaction rates without performing a full load.
    
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

    rate_dict = {} #Dictionary to store summed rates.
    keep = lambda v,vals,cast=lambda s:s : (vals is None) or (v in vals) or (cast(v) in vals)
    keep_time = lambda t: (times is None) or (np.abs(np.nanmin(np.log10(float(t))-np.log10(times))) < 1e-3)
    npath = 0
    for fpath in fpaths:
        npath+=1
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

################################################################################
############################ Read Lambda Files #################################
################################################################################
def read_levels(fpath):
    f = open(fpath)
    line = f.readline()
    
    #Find start of levels section
    row = 0
    while not '!LEVEL' in line: 
        line = f.readline()
        row+=1
    
    dat = []
    line = f.readline()
    row+=1
    start = row
    print(start)
    while not '!' in line:
        line = f.readline()
        row+=1
    nrows=row-start

    dat = np.genfromtxt(fpath,comments='!',skip_header=start,max_rows=nrows,dtype=float)
    return dat

def read_trans(fpath):
    f = open(fpath)
    line = f.readline()
    
    #Find start of levels section
    row = 0
    while not '!TRANS' in line: 
        line = f.readline()
        row+=1
    
    dat = []
    line = f.readline()
    row+=1
    start = row
    print(start)
    while not '!' in line:
        line = f.readline()
        row+=1
    nrows=row-start

    dat = np.genfromtxt(fpath,comments='!',skip_header=start,max_rows=nrows,dtype=float)
    return dat