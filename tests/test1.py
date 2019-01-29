from .. import chem_mod
import os

print(os.getcwd())


env_path = 'chem_mod/tests/test_model/environ/test_environ/test_mod/'
inp_file = '0io.isrf4_test.inp'
out_path = 'chem_mod/tests/test_model/runs/test/'

cmod = chem_mod(env_path,inp_file,out_path)
print("Successfully loaded chem_mod instance.")
