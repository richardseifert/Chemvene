import numpy as np
import glob
import time

def find_mol(fpaths, strmol):
    strmol = ' '+strmol.lstrip().rstrip()+' ' #Pad strmol with spaces to avoid similar molecule names.
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
    #Read lines from file.
    f = open(fpath)
    lines = f.read().split('\n')
    f.close()

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

def load_mol(fpath,strmol,mol_head=None,nrows=None,col=None):
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
    print(direc+'r*.out')
    
    #Get location of strmol in r.out files, and timesteps.
    mol_start,nrows,col,Times = find_mol(fpaths,strmol)
    nTimes = len(Times)

    #Get list of rad.
    fpaths = glob.glob(direc+"r*_1.out")
    radnam = np.array([fpath.split("/")[-1].split("_e1")[0][1:] for fpath in fpaths])
    radval = np.array([float(strg.rstrip()) for strg in radnam])
    #Sort by radius.
    sort = np.argsort(radval)
    radnam = radnam[sort]
    radval = radval[sort]
    nR = len(radnam)

    # Find maximum z value from filenames of .out files.
    fpaths = glob.glob(direc+"r"+radnam[0]+"*.out")
    zval = [int(fpath.split("/")[-1].split("_")[-1].split(".")[0]) for fpath in fpaths]
    min_z = min(zval)
    nZ = len(zval)

    mol_start,nrows,col,Times = find_mol(fpaths,strmol)
    nTimes = len(Times)

    dat = np.zeros((nR*nZ*nTimes,4))
    fpaths = glob.glob(direc+'r*.out')
    row_i = 0
    t = time.time()
    for i,path in enumerate(fpaths):
        #rad = float(path.split('/')[-1].split('_e1')[0][1:])
        rau,zau,Tg,Td,rho = load_physical(path) 
        z = int(path.split('/')[-1].split('_')[-1].split('.')[0])
        ab = load_mol(path,strmol,mol_start,nrows,col)
        row_f = row_i+nTimes
        dat[row_i:row_f,0] = Times
        dat[row_i:row_f,1] = rau
        dat[row_i:row_f,2] = z
        if z == min_z:
            dat[row_i:row_f,3] = 0.
        else:
            dat[row_i:row_f,3] = ab
        row_i = row_f
    
    return dat 

def load_physical(fpath):
    f = open(fpath)
    line = f.readline()
    while not "INITIAL VALUES:" in line:
        line = f.readline()
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