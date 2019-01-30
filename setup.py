from setuptools import setup

setup(name='chem_mod',
      version='0.1',
      description='A package for handling output from our chemical code.',
      url='https://github.com/richardseifert/chem_mod',
      author='Richard Seifert',
      author_email='seifertricharda@gmail.com',
      license='MIT',
      packages=['chem_mod','chem_mod.read','chem_mod.pkg_files'],
      zip_safe=False,
      include_package_data=True)
