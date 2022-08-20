cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 

#@cython.boundscheck(False)
#@cython.wraparound(False)

cdef int find_min_freq(double * freqs,double lower_freq,int num_freqs):
	cdef int i
	if freqs[0]>=lower_freq:
		return 0
	for i in range(1,num_freqs):
		if freqs[i]==lower_freq:
			return i
		if freqs[i]>lower_freq and freqs[i-1]<lower_freq:
			return i

cdef int find_max_freq(double * freqs,double upper_freq,int num_freqs):
	cdef int i
	for i in range(num_freqs-1):
		if freqs[i]==upper_freq:
			return i
		if freqs[i]<upper_freq and freqs[i+1]>upper_freq:
			return i
	if freqs[num_freqs-1]<=upper_freq:
		return num_freqs-1		

cdef int detect_low_snr_freqs(double *spectrum,double *rms, double rms_thresh, int *pos, int num_freqs):
	cdef int i
	cdef int j
	j=0
	for i in range(num_freqs):
		if rms[i]<1e-5:
			pos[i]=1
			j=j+1
		elif spectrum[i]<rms_thresh*rms[i]:
			pos[i]=1
			j=j+1
		else:
			pos[i]=0
	return j

cdef double square(double x):
	return x*x	
	
cdef void fill_value(double [:] model, double * model_comb, int num_freqs):	
	cdef int i
	for i in range(num_freqs):
		model_comb[i]=model[i]
		
	return

cdef double min_chi_square(double [:]  model,\
		double *spectrum,double *error,int low_ind,int high_ind, \
		double rms_thresh, double *rms, int num_params, int num_freqs, \
		int *param_lengths, int *low_snr_freqs,\
		double sys_error, int * param_inds):
	
	cdef double min_chi=1e100
	cdef int i,j,min_ind
	cdef double chi=0.0
	
	
	cdef int num_elem_model,num_param_comb=1
	
	for i in range(num_params):
		num_param_comb=num_param_comb*param_lengths[i]
	num_elem_model=num_param_comb*num_freqs
	cdef int k=0
	
	i=0
	while i<num_elem_model:
	#for i in range(0,num_elem_model,num_freqs):
		chi=0.0
		for j in range(low_ind,high_ind+1):
			if low_snr_freqs[j]==0:
				chi=chi+square((spectrum[j]-model[i+j])/error[j])
			else:
				if model[i+j]>(1+sys_error)*rms_thresh*rms[j]:
					chi=chi+1000   ### above the upper limit. Hence a high value to the chi square
				else:
					chi=chi+0.0
		
		if chi<min_chi:
			min_chi=chi
			min_ind=k
		k=k+1
		i=i+num_freqs
	
	
	for i in range(num_params-1,-1,-1):
		param_inds[i]=min_ind%param_lengths[i]
		min_ind=(min_ind-param_inds[i])/param_lengths[i]
	
	return min_chi		
		
		



cpdef numpy.ndarray[numpy.double_t,ndim=1] compute_min_chi_square(numpy.ndarray[numpy.double_t, ndim=1] model, \
				numpy.ndarray[numpy.double_t,ndim=4] cube,\
				numpy.ndarray[numpy.double_t,ndim=2] err_cube,\
				numpy.ndarray[numpy.double_t,ndim=3] lower_freq,\
				numpy.ndarray[numpy.double_t,ndim=3] upper_freq,\
				numpy.ndarray[numpy.int_t,ndim=1] param_lengths,\
				numpy.ndarray[numpy.double_t,ndim=1] freqs,
				double sys_error, double rms_thresh, int min_freq_num, int num_params,\
				int num_times, int num_freqs, int num_y,int num_x):
				
	cdef int t,y1,x1,i,j,l
	cdef numpy.ndarray[numpy.double_t,ndim=1] fitted=np.zeros(num_times*num_y*num_x*(num_params+1))
	cdef int low_ind, high_ind
	cdef double *spectrum
	cdef double *rms
	cdef double *sys_err
	cdef double *error
	
	spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	sys_err=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	error=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	cdef int *pos
	pos=<int *>PyMem_Malloc(num_freqs*sizeof(int))
	
	cdef int *param_ind
	param_ind=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	cdef int low_snr_freq_num
	cdef double red_chi
	
	cdef int *param_lengths1
	param_lengths1=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	cdef double *freqs1
	freqs1=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	for i in range(num_params):
		param_lengths1[i]=param_lengths[i]  #### making it a raw pointer. Faster access
	
	for i in range(num_freqs):
		freqs1[i]=freqs[i]
		
	for t in range(num_times):
		for i in range(num_freqs):
			rms[i]=err_cube[t,i]
		for y1 in range(num_y):
			for x1 in range(num_x):
				low_ind=find_min_freq(freqs1,lower_freq[t,y1,x1],num_freqs)
				high_ind=find_max_freq(freqs1,upper_freq[t,y1,x1],num_freqs)
				for j in range(num_freqs):
					spectrum[j]=cube[t,j,y1,x1]
					sys_err[j]=sys_error*spectrum[j]
					error[j]=sqrt(rms[j]**2+sys_err[j]**2)
					
				
				low_snr_freq_num=detect_low_snr_freqs(spectrum,rms,rms_thresh,pos,num_freqs)
				if low_snr_freq_num>min_freq_num:
					for l in range(num_params):
						fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l]=-1
					fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params]=-1
					continue
				
				red_chi=min_chi_square(model,spectrum,error,low_ind,high_ind,\
						rms_thresh,rms,num_params,num_freqs,param_lengths1,pos,sys_error,param_ind)
				
				for l in range(num_params):
					fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l]=param_ind[l]
				fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params]=red_chi
				
			
	PyMem_Free(pos)
	PyMem_Free(param_ind)
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(sys_err)
	PyMem_Free(error)
	return fitted
				
