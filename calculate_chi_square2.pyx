cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython
#@cython.boundscheck(False)
#@cython.wraparound(False)


cdef double square(double x):
	return x*x

cdef double min_chi_square(double [:]  model,\
		double *spectrum,double *error,int low_ind,int high_ind, \
		double rms_thresh, double *rms, int num_params, int num_freqs, \
		int *param_lengths, int *low_snr_freqs,\
		double sys_error, int * param_inds):
	
	cdef double min_chi=1e100
	cdef int i,j,min_ind
	cdef double chi=0.0
	cdef int k=0
	cdef int mid_ind
	cdef double ratio
	
	cdef int num_elem_model,num_param_comb=1
	
	for i in range(num_params):
		num_param_comb=num_param_comb*param_lengths[i]
	num_elem_model=num_param_comb*num_freqs
	cdef int param1=0
	
	i=0
	while i<num_elem_model:
	#for i in range(0,num_elem_model,num_freqs):
		chi=0.0
		mid_ind=(low_ind+high_ind)//2
		ratio=spectrum[mid_ind]/model[i+mid_ind]
		if ratio>3 or ratio<0.3:  ## I added this line to make the code faster.
					   ### The argument is that of at the mid-freq, the
					   ### spectrum is unlikely to be fit well by the model. 
			k=k+1
			i=i+num_freqs
			continue
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
		min_ind=(min_ind-param_inds[i])//param_lengths[i]
	
	return min_chi	

cdef void calc_grad(double [:]fitted, int num_times, int num_y, int num_x, int t, int y1, int x1, int num_params,\
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

cdef double calc_grad_chi(double [:]fitted, int num_times, int num_y, int num_x, int num_params, \
				double *grad_x, double *grad_y, double *grad_t,\
				double spatial_smoothness_enforcer=0.0, double temporal_smoothness_enforcer=0.0):
	cdef double tot_chi=0.0
	cdef int t,y1,x1,ind,param1
	cdef int ind1,ind2,ind3
	
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				ind3=t*num_y*num_x*num_params+y1*num_x*num_params+x1*num_params
				
				if fitted[ind]>-0.2:
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

cdef double calc_red_chi_all_pix(int num_times, int num_freqs, int num_y, int num_x, int num_params,int ***low_freq_ind,\
						int ***upper_freq_ind, numpy.ndarray[numpy.double_t,ndim=4] cube,\
						numpy.ndarray[numpy.double_t,ndim=2] err_cube,numpy.ndarray[numpy.double_t,ndim=1] model, double sys_error,\
						int *pos,double [:] fitted, double *freqs1, double lower_freq,\
						double upper_freq, int first_pass, double rms_thresh, int min_freq_num, int *param_lengths1,\
						int *param_ind, double spatial_smoothness_enforcer,\
						double temporal_smoothness_enforcer):
						
						
	cdef double *spectrum
	cdef double *rms
	cdef double *sys_err
	cdef double *error
	cdef double red_chi
	cdef int low_ind,high_ind, t, x1,y1,j, ind3,ind
	cdef int low_snr_freq_num
	cdef int ind5
	
	spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	sys_err=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	error=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	for i in range(num_freqs):
		spectrum[i]=0.0
		rms[i]=0.0
		sys_err[i]=0.0
		error[i]=0.0
	
			
	for t in range(num_times):
		for i in range(num_freqs):
			rms[i]=err_cube[t,i]
		for y1 in range(num_y):
			for x1 in range(num_x):
				for j in range(num_freqs):
					spectrum[j]=cube[t,j,y1,x1]
					sys_err[j]=sys_error*spectrum[j]
					error[j]=sqrt(rms[j]**2+sys_err[j]**2)
				
				ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				if fitted[ind5]<-0.2:
					continue
				ind3=t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs	
				red_chi=min_chi_square(model,spectrum,error,low_freq_ind[t][y1][x1],upper_freq_ind[t][y1][x1],\
							rms_thresh,rms,num_params,num_freqs,param_lengths1,pos+ind3,sys_error,param_ind)
				
				for l in range(num_params):
					ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l
					fitted[ind5]=param_ind[l]
				ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				fitted[ind5]=red_chi	
					
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(sys_err)
	PyMem_Free(error)
	
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
					
	cdef double grad_chi=calc_grad_chi(fitted,num_times,num_y,num_x,num_params, \
						grad_x,grad_y,grad_t, spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
	return grad_chi
