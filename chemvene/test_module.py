import glob

from .read import ensure_pad, find_mol_helper, find_mol, load_physical
from . import __path__ as pkg_path
from . import chem_mod

test_base_dir = pkg_path[0]+'/test_files/'
test_model = test_base_dir+'test_model/'

class TestRead:
    ######################### Test Abundance Reading ##########################

    ### Define global testing variables! ###
    out_fpaths = glob.glob(test_model+'e1/*.out')
    
    
    ### Test ensure_pad ###
    def test_ensure_pad(self):
        assert ensure_pad('H') == ' H '
        assert ensure_pad('H H') == ' H H '
        assert ensure_pad(' H  ') == ' H '
        assert ensure_pad('  H') == ' H '
        assert ensure_pad(' Howdy ') == ' Howdy '

    ### Test find_mol_helper ###
    def template_find_mol_helper(self,fi,strmol,c_mol_head=None,c_nrows=None,c_col=None):
        out_fpath = self.out_fpaths[fi]
        mol_head,nrows,col,times = find_mol_helper(out_fpath,strmol)
        assert mol_head == c_mol_head
        assert col == c_col
        assert nrows == c_nrows
        assert len(times) == nrows
    def test_find_mol_helper1(self):
        self.template_find_mol_helper(0,'H3+',55,10,5)
    def test_find_mol_helper2(self):
        self.template_find_mol_helper(3,'H3+',55,10,5)
    def test_find_mol_helper3(self):
        self.template_find_mol_helper(9,'H3+',55,10,5)
    def test_find_mol_helper4(self):
        self.template_find_mol_helper(6,'H',39,10,1)
    def test_find_mol_helper5(self):
        self.template_find_mol_helper(-5,'He+',55,10,6)
    def test_find_mol_helper6(self):
        self.template_find_mol_helper(3,'E',39,10,4)
    def test_find_mol_helper7(self):
        self.template_find_mol_helper(2,'H-',71,10,1)

    ### Test find_mol ###
    def template_find_mol(self,strmol,c_mol_head=None,c_nrows=None,c_col=None):
        mol_head,nrows,col,times = find_mol(self.out_fpaths,strmol)
        assert mol_head == c_mol_head
        assert col == c_col
        assert nrows == c_nrows
        assert len(times) == nrows
    def test_find_mol1(self):
        self.template_find_mol('GRAIN0',39,10,6)
    def test_find_mol2(self):
        self.template_find_mol('H2+',55,10,4)
    def test_find_mol3(self):
        self.template_find_mol('H3+',55,10,5)
    def test_find_mol4(self):
        self.template_find_mol('H',39,10,1)
    def test_find_mol5(self):
        self.template_find_mol('He+',55,10,6)
    def test_find_mol6(self):
        self.template_find_mol('E',39,10,4)
    def test_find_mol7(self):
        self.template_find_mol('H-',71,10,1)

    ### Test load_physical ###
    def template_load_physical(self,out_fpath,c_rau,c_height,c_Tg,c_Td,c_rho):
        rau,height,Tg,Td,rho = load_physical(out_fpath)
        assert    rau == c_rau
        assert height == c_height
        assert     Tg == c_Tg
        assert     Td == c_Td
        assert    rho == c_rho
    def test_load_physical1(self):
        out_fpath = test_model+'e1/r32.0981_e1_16.out'
        self.template_load_physical(out_fpath,3.21e1,3.276e0,27.8,27.8,7.462e-14)
    def test_load_physical2(self):
        out_fpath = test_model+'e1/r10.6547_e1_8.out'
        self.template_load_physical(out_fpath,1.065e1,2.885e0,91.6,85.4,4.447e-15)
    def test_load_physical3(self):
        out_fpath = test_model+'e1/r10.6547_e1_16.out'
        self.template_load_physical(out_fpath,1.065e1,1.087e0,37.0,37.0,7.101e-13)
    def test_load_physical4(self):
        out_fpath = test_model+'e1/r20.0087_e1_4.out'
        self.template_load_physical(out_fpath,2.001e1,7.208e0,109.5,85.4,3.529e-16)
    def test_load_physical5(self):
        out_fpath = test_model+'e1/r32.0981_e1_9.out'
        self.template_load_physical(out_fpath,3.21e1,7.993e0,42.2,42.1,8.577e-15)

class TestChemMod:
    #Load test model for tests
    cmod = chem_mod(test_model,base_dir=test_base_dir)

    ### Test get_quant and helpers ###
    ## Test quant validators
    def test_validate_phys_true(self):
        assert self.cmod._validate_phys('rho') == True
    def test_validate_phys_false(self):
        assert self.cmod._validate_phys('spaghetti') == False
    def test_validate_abun_true(self):
        assert self.cmod._validate_abun('H3+',time=0) == True
    def test_validate_abun_false(self):
        assert self.cmod._validate_abun('meatballs',time=0) == False
    def test_validate_dens_true(self):
        assert self.cmod._validate_abun('nH(gr)',time=0) == True
    def test_validate_dens_false(self):
        assert self.cmod._validate_abun('nmeatballs',time=0) == False
    def test_validate_radf_true(self):
        assert self.cmod._validate_radf('uv') == True
    def test_validate_radf_none(self):
        assert self.cmod._validate_radf('isrf') == False
    def test_validate_radf_false(self):
        assert self.cmod._validate_radf('tomatosauce') == False