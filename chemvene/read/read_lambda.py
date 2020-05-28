import numpy as np

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
