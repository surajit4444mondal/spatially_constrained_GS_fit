cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython
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
		double sys_error, int * param_inds, double *grad_x,\
		double *grad_y, double *grad_t, double spatial_smoothness_enforcer,\
		double temporal_smoothness_enforcer, int num_x, int num_y, int num_times):
	
	cdef double min_chi=1e100
	cdef int i,j,min_ind
	cdef double chi=0.0
	cdef int k
	
	cdef int num_elem_model,num_param_comb=1
	
	for i in range(num_params):
		num_param_comb=num_param_comb*param_lengths[i]
	num_elem_model=num_param_comb*num_freqs
	cdef int param1=0
	
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
		if spatial_smoothness_enforcer>0:
			if num_x>1:
				for param1 in range(num_params):
					chi=chi+spatial_smoothness_enforcer*square(grad_x[param1])
			if num_y>1:
				for param1 in range(num_params):
					chi=chi+spatial_smoothness_enforcer*square(grad_y[param1])
			
			if num_times>1:
				for param1 in range(num_params):
					chi=chi+temporal_smoothness_enforcer*square(grad_t[param1])
					
		if chi<min_chi:
			min_chi=chi
			min_ind=k
		k=k+1
		i=i+num_freqs
	
	
	for i in range(num_params-1,-1,-1):
		param_inds[i]=min_ind%param_lengths[i]
		min_ind=(min_ind-param_inds[i])/param_lengths[i]
	
	return min_chi	
	
cdef int find_max_pos(double *spectrum, int *lower_freq_ind, int *upper_freq_ind, int num_freqs):
	cdef int i, max_loc
	cdef double max_val=-1.0
	for i in range(lower_freq_ind[0],upper_freq_ind[0]+1):
		if spectrum[i]>max_val:
			max_loc=i
			max_val=spectrum[i]
	return max_loc
			
		
cdef void calc_fitrange_homogenous(double *spectrum, int *lower_freq_ind, int *upper_freq_ind,int num_freqs):
	cdef int max_loc=find_max_pos(spectrum,lower_freq_ind, upper_freq_ind, num_freqs)		
	cdef double *smoothed_spectrum
	smoothed_spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	cdef int smooth_length=3
	cdef int i,j
	cdef int smooth_interval=smooth_length//2
	cdef double sum1
	
	for i in range(lower_freq_ind[0],upper_freq_ind[0]):
		if i<smooth_interval:
			smoothed_spectrum[i]=0.0
			continue
		elif i>num_freqs-1-smooth_interval:
			smoothed_spectrum[i]=0.0
			continue
		j=-smooth_interval
		sum1=0.0
		while j<=smooth_interval:
			sum1=sum1+spectrum[i+j]
			j=j+1
		sum1=sum1/smooth_length
		smoothed_spectrum[i]=sum1
	

	j=max_loc-smooth_interval
	
	
	while j>smooth_interval:
		if smoothed_spectrum[j]<1e-7:
			lower_freq_ind[0]=j
			break
		if (smoothed_spectrum[j-1]-smoothed_spectrum[j])/smoothed_spectrum[j]>0.05:
			lower_freq_ind[0]=j
			break
		j=j-1
		
	j=max_loc+1+smooth_interval
	upper_freq_ind[0]=j
	
	while j<num_freqs-2-smooth_interval:
		if smoothed_spectrum[j]<1e-7:
			upper_freq_ind[0]=j
			break
		if (smoothed_spectrum[j+1]-smoothed_spectrum[j])/smoothed_spectrum[j]>0.05:
			upper_freq_ind[0]=j
			break
		j=j+1
		
	PyMem_Free(smoothed_spectrum)
	return
	
cdef void calc_grad(double *fitted, int num_times, int num_y, int num_x, int t, int y1, int x1, int num_params,\
					double * grad_x, double * grad_y, double * grad_t):


	cdef int ind,ind1, ind2,param1
	
	if num_x>1:
		if x1>=1 and x1<=num_x-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1-1)*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1+1)*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(ind2-ind1)/2
				elif fitted[ind1-param1+num_params]>0:
					grad_x[param1]=(ind-ind1)
				elif fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(ind2-ind)
				else:
					grad_x[param1]=0.0
		elif x1==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1+1)*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(ind2-ind)
				else:
					grad_x[param1]=0.0
		elif x1==num_x-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1-1)*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_x[param1]=(ind-ind1)
				else:
					grad_x[param1]=0.0
	if num_y>1:
		if y1>=1 and y1<=num_y-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+(y1-1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+(y1+1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(ind2-ind1)/2
				elif fitted[ind1-param1+num_params]>0:
					grad_y[param1]=(ind-ind1)
				elif fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(ind2-ind)
				else:
					grad_y[param1]=0.0
		elif y1==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+(y1+1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(ind2-ind)
				else:
					grad_y[param1]=0.0
		elif y1==num_y-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+(y1-1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_y[param1]=(ind-ind1)
				else:
					grad_y[param1]=0.0
	
	if num_times>1:
		if t>=1 and t<=num_times-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=(t-1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=(t+1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(ind2-ind1)/2
				elif fitted[ind1-param1+num_params]>0:
					grad_t[param1]=(ind-ind1)
				elif fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(ind2-ind)
				else:
					grad_t[param1]=0.0
		elif t==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=(t+1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(ind2-ind)
				else:
					grad_t[param1]=0.0
		elif t==num_times-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=(t-1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_t[param1]=(ind-ind1)
				else:
					grad_t[param1]=0.0

cdef double calc_grad_chi(double *fitted, int num_times, int num_y, int num_x, int num_params, \
				double *grad_x, double *grad_y, double *grad_t,\
				double spatial_smoothness_enforcer=0.0, double temporal_smoothness_enforcer=0.0, int max_iter=10):
	cdef double tot_chi=0.0
	cdef int t,y1,x1,ind,param1
	cdef int ind1,ind2,ind3
	
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				ind3=t*num_y*num_x*num_params+y1*num_x*num_params+x1*num_params
				if fitted[ind]>0:
					tot_chi=tot_chi+fitted[ind]
					calc_grad(fitted,num_times,num_y,num_x,t,y1,x1,num_params,grad_x+ind3,grad_y+ind3,grad_t+ind3)
					if num_x>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+spatial_smoothness_enforcer*square(grad_x[ind3+param1])
					if num_y>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+spatial_smoothness_enforcer*square(grad_y[ind3+param1])
					
					if num_times>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+temporal_smoothness_enforcer*square(grad_t[ind3+param1])
							
	return tot_chi
	
cdef calc_red_chi_all_pix(int num_times, int num_freqs, int num_y, int num_x, int num_params,int ***low_freq_ind,\
						int ***upper_freq_ind, numpy.ndarray[numpy.double_t,ndim=4] cube,\
						numpy.ndarray[numpy.double_t,ndim=2] err_cube,numpy.ndarray[numpy.double_t,ndim=1] model, double sys_error,\
						int *pos,double *fitted, double *freqs1, double lower_freq,\
						double upper_freq, int first_pass, double rms_thresh, int min_freq_num, int *param_lengths1,\
						int *param_ind, double *grad_x, double *grad_y, double *grad_t, double spatial_smoothness_enforcer,\
						double temporal_smoothness_enforcer):
						
						
	cdef double *spectrum
	cdef double *rms
	cdef double *sys_err
	cdef double *error
	cdef double red_chi
	cdef int low_ind,high_ind, t, x1,y1,j, ind3
	cdef int low_snr_freq_num
	
	spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	sys_err=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	error=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	
	for t in range(num_times):
		for i in range(num_freqs):
			rms[i]=err_cube[t,i]
		for y1 in range(num_y):
			for x1 in range(num_x):
				ind3=t*num_y*num_x*num_params+y1*num_x*num_params+x1*num_params
				if first_pass==0:
					low_ind=find_min_freq(freqs1,lower_freq,num_freqs)
					high_ind=find_max_freq(freqs1,upper_freq,num_freqs)
				for j in range(num_freqs):
					spectrum[j]=cube[t,j,y1,x1]
					sys_err[j]=sys_error*spectrum[j]
					error[j]=sqrt(rms[j]**2+sys_err[j]**2)
				if first_pass==0:
					calc_fitrange_homogenous(spectrum, &low_ind, &high_ind,num_freqs)
					low_freq_ind[t][y1][x1]=low_ind
					upper_freq_ind[t][y1][x1]=high_ind
					low_snr_freq_num=detect_low_snr_freqs(spectrum,rms,rms_thresh,pos,num_freqs)
					if low_snr_freq_num>min_freq_num:
						for l in range(num_params):
							fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l]=-1
						fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params]=-1
						continue
				if first_pass!=0:	
					red_chi=min_chi_square(model,spectrum,error,low_freq_ind[t][y1][x1],upper_freq_ind[t][y1][x1],\
							rms_thresh,rms,num_params,num_freqs,param_lengths1,pos,sys_error,param_ind, grad_x+ind3,grad_y+ind3,grad_t+ind3,\
							spatial_smoothness_enforcer, temporal_smoothness_enforcer, num_x, num_y, num_times)
				else:
					red_chi=min_chi_square(model,spectrum,error,low_freq_ind[t][y1][x1],upper_freq_ind[t][y1][x1],\
							rms_thresh,rms,num_params,num_freqs,param_lengths1,pos,sys_error,param_ind, grad_x+ind3,grad_y+ind3,grad_t+ind3,\
							-1,-1, num_x, num_y, num_times)
				
				for l in range(num_params):
					fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l]=param_ind[l]
				fitted[t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params]=red_chi
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(sys_err)
	PyMem_Free(error)
	return		
							
cpdef numpy.ndarray[numpy.double_t,ndim=1] compute_min_chi_square(numpy.ndarray[numpy.double_t, ndim=1] model, \
				numpy.ndarray[numpy.double_t,ndim=4] cube,\
				numpy.ndarray[numpy.double_t,ndim=2] err_cube,\
				double lower_freq,\
				double upper_freq,\
				numpy.ndarray[numpy.int_t,ndim=1] param_lengths,\
				numpy.ndarray[numpy.double_t,ndim=1] freqs,
				double sys_error, double rms_thresh, int min_freq_num, int num_params,\
				int num_times, int num_freqs, int num_y,int num_x, double spatial_smoothness_enforcer,\
				double temporal_smoothness_enforcer):
				
	cdef int t,y1,x1,i,j,l,ind
	cdef numpy.ndarray[numpy.double_t,ndim=1] fitted1=np.zeros(num_times*num_y*num_x*(num_params+1))
	cdef double *fitted
	fitted=<double *>PyMem_Malloc(num_times*num_y*num_x*(num_params+1)*sizeof(double))
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
	
	cdef int ***low_freq_ind
	cdef int ***upper_freq_ind
	
	low_freq_ind=<int ***>PyMem_Malloc(num_times*sizeof(int ***))
	upper_freq_ind=<int ***>PyMem_Malloc(num_times*sizeof(int ***))
	
	for t in range(num_times):
		low_freq_ind[t]=<int **>PyMem_Malloc(num_y*sizeof(int **))
		upper_freq_ind[t]=<int **>PyMem_Malloc(num_y*sizeof(int **))
		for y1 in range(num_y):
			low_freq_ind[t][y1]=<int *>PyMem_Malloc(num_x*sizeof(int))
			upper_freq_ind[t][y1]=<int *>PyMem_Malloc(num_x*sizeof(int))
	
	for i in range(num_params):
		param_lengths1[i]=param_lengths[i]  #### making it a raw pointer. Faster access
	
	for i in range(num_freqs):
		freqs1[i]=freqs[i]
	
	
	cdef int first_try=0	
	
	
	
	cdef double *grad_x
	cdef double *grad_y
	cdef double *grad_t
	grad_x=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	grad_y=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	grad_t=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				for l in range(num_params):
					ind=t*num_y*num_x*num_params+y1*num_x*num_params+x1*num_params+l
					grad_x[ind]=0.0
					grad_y[ind]=0.0
					grad_t[ind]=0.0
					
	calc_red_chi_all_pix(num_times, num_freqs, num_y, num_x, num_params,low_freq_ind,\
						upper_freq_ind, cube, err_cube, model, sys_error,\
						pos, fitted, freqs1, lower_freq,\
						upper_freq, first_try, rms_thresh, min_freq_num,\
						param_lengths1,param_ind, grad_x,grad_y,grad_t,\
						spatial_smoothness_enforcer, temporal_smoothness_enforcer)
	
	cdef double grad_chi=calc_grad_chi(fitted,num_times,num_y,num_x,num_params, \
						grad_x,grad_y,grad_t, spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
	print (grad_chi)
	cdef int iterations=1
	
	#while iteration<=max_iter:
		
	
	
						
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				for l in range(num_params+1):
					ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l
					fitted1[ind]=fitted[ind]
	
	
	for t in range(num_times):
		for y1 in range(num_y):
			PyMem_Free(low_freq_ind[t][y1])
			PyMem_Free(upper_freq_ind[t][y1])
		PyMem_Free(low_freq_ind[t])
		PyMem_Free(upper_freq_ind[t])	
	PyMem_Free(low_freq_ind)
	PyMem_Free(upper_freq_ind)	
			
	PyMem_Free(pos)
	PyMem_Free(param_ind)
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(sys_err)
	PyMem_Free(error)
	PyMem_Free(fitted)
	PyMem_Free(grad_x)
	PyMem_Free(grad_y)
	PyMem_Free(grad_t)
	return fitted1
				
