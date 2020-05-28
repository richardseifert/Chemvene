def combine(cmod1,cmod2,mask):
    new_cmod = cmod1.copy()
    new_cmod.set_all(cmod2,mask) 
    return new_cmod
