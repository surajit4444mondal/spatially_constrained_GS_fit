from setuptools import setup, Extension
import numpy
#from Cython.Build import cythonize

#ext_modules=[Extension('calculate_chi_square2',sources=["calculate_chi_square2.pyx"],libraries=["m"],include_dirs=[numpy.get_include()]),\
#		 Extension('spatial_minimiser',sources=["spatial_minimiser.pyx"],libraries=["m"],include_dirs=[numpy.get_include()])]

ext_modules=[Extension('spatial_minimiser1',sources=["spatial_minimiser1.pyx"],libraries=["m"],include_dirs=[numpy.get_include()])]
for e in ext_modules:
	e.cython_directives = {'language_level': "3"} #all are Python-3


setup(setup_requires=['setuptools>18.0','cython'],ext_modules=ext_modules)
