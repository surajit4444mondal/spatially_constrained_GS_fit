import cython
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.time cimport time,time_t



cpdef double gen_seed():
	cdef timespec ts
	cdef double current
	cdef time_t t#,current
	cdef int i
	cdef int num=10
	cdef unsigned int a=8121
	cdef unsigned int c=28411
	cdef unsigned int m=134456
	cdef unsigned int x,x1

	t = time(NULL) 
	clock_gettime(CLOCK_REALTIME, &ts)
	current = ts.tv_sec + (ts.tv_nsec / 1000000000.)-t
	
	
	for i in range(num):
		x=int(current*a)+c#a*current+c
		x=x%m
		print (x/134456)
		current=x
	
	return current
	
	
