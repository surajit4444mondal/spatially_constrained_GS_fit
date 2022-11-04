from setuptools import setup, Extension
import numpy


ext_modules=[Extension('cython_functions_for_fast_computation',sources=["cython_functions_for_fast_computation.pyx"],libraries=["m"],include_dirs=[numpy.get_include()])]
for e in ext_modules:
	e.cython_directives = {'language_level': "3"} #all are Python-3


setup(setup_requires=['setuptools>18.0','cython'],ext_modules=ext_modules)
