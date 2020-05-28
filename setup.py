from setuptools import setup

setup(name='chemvene',
      version='0.1',
      description='A convenient chemical model visualization and analysis engine for processing output from the Michigan protoplanetary disk chemical modeling code.',
      url='https://github.com/richardseifert/Chemvene',
      author='Richard Seifert',
      author_email='seifertricharda@gmail.com',
      license='MIT',
      packages=['chemvene','chemvene.read','chemvene.pkg_files'],
      install_requires=[
            'matplotlib',
            'numpy',
            'pandas',
            'scipy',

      ],
      zip_safe=False,
      include_package_data=True)
