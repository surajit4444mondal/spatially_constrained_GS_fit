cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython

cpdef void test_function(double[::1]x, double[::1]y, int num):
	cdef double *x1
	#x1=<double *>PyMem_Malloc(10*sizeof(double))
	
	cdef double *y1
	#y1=<double *>PyMem_Malloc(10*sizeof(double))	
	
	cdef int i
	
	x1=&x[0]
	y1=&y[0]
	
	for i in range(num):
		x1[i]=i*1.2
		y1[i]=i/2
	
	
	return
