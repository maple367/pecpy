from setuptools import setup
# from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

# Get the requirements from the relevant file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='pecpy',
      version='0.7',
      description='Optimization-based proximity effect correction',
      long_description=long_description,
      url='https://git.kern.phys.au.dk/SunTune/pecpy',
      author='Emil Haldrup Eriksen',
      author_email='emher@au.dk',
      license='MIT',
      install_requires=required,
      include_package_data=True,
      packages=['pecpy'],
      zip_safe=False)
