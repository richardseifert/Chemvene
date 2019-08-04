import numpy as np

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
        specv = next(filter(None,line.split(' ')))
        try:
            spec_vals.append(float(specv))
        except ValueError:
            raise ValueError('Expected float %s value, got %s'%(specq,specv))
        line = f.readline()
    
    #Rewind to start of file.
    f.seek(0)

    dat = np.empty((4,0))
    f.readline() #First line is garbage
    line=f.readline()
    while not len(line)==0:
        info = list(filter(None,line.split(' ')))
        assert (info[0]=='Radius(AU)'),'Expected radius line, got %s'%(line)
        R = float(info[1])
        line=f.readline()
        info = list(filter(None,line.split(' ')))
        assert (info[0]=='z(AU)'),'Expected height z line, got %s'%(line)
        z_arr = np.array(info[1:]).astype(float)
        R_arr = R*np.ones_like(z_arr) 
        f.readline() #Ignore spec/flux unit line.
        #Go through flux lines
        line = f.readline()
        while line.split(' ')[0]!='Radius(AU)' and len(line)>0:
            info = list(filter(None,line.split(' ')))
            specv = float(info[0])
            flux_arr = np.array(info[1:]).astype(float)
            spec_arr = specv*np.ones_like(flux_arr) 
            arr = np.array([R_arr,z_arr,spec_arr,flux_arr])
            dat = np.append(dat,arr,axis=1)
            line = f.readline()
    return dat.T
