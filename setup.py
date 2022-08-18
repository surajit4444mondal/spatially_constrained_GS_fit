from setuptools import setup, Extension
import numpy
#from Cython.Build import cythonize

setup(
    setup_requires=['setuptools>18.0','cython'],
    ext_modules = [Extension('calculate_chi_square',sources=["calculate_chi_square.pyx"],libraries=["m"],include_dirs=[numpy.get_include()])]
)
#python3 setup.py build_ext --inplace
