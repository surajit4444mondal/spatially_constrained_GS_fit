cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython
import h5py
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.time cimport time,time_t
#@cython.boundscheck(False)
#@cython.wraparound(False)


cdef int find_high_indx(int high_ind_val, int num):
	cdef int i
	i=high_ind_val
	while i>=num:
		i=i-1
	return i

cdef int find_low_indx(int low_ind_val):
	cdef int i=low_ind_val
	while i<0:
		i=i-1
	return i
	
cdef int check_present_already(int *param_val_ind,int value, int max_size):
	cdef int i
	for i in range(max_size):
		if param_val_ind[i]==value:
			return -1
		if param_val_ind[i]<0:
			return i
	return 0
	
cdef void gen_seed(double *seed, int num):
	cdef timespec ts
	cdef double current
	cdef time_t t
	cdef int i
	cdef unsigned int a=8121
	cdef unsigned int c=28411
	cdef unsigned int m=134456
	cdef unsigned int x

	t = time(NULL) 
	clock_gettime(CLOCK_REALTIME, &ts)
	current = ts.tv_sec + (ts.tv_nsec / 1000000000.)-t
	
	
	for i in range(num):
		x=int(current*a)+c
		x=x%m
		seed[i]=x/134456.0
		current=x
	return	
	
cdef int find_model_ind(double [:] fitted, int x, int y, double *param_lengths, int num_params,\
			int num_x, int num_y):
	
	cdef int i,ind,j, product
	ind=y*num_x*(num_params+1)+x*(num_params+1)
	
	cdef int *param_inds
	param_inds=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	for i in range(num_params):
		param_inds[i]=fitted[ind+i]
	
	
	for i in range(num_params):
		product=1
		for j in range(i+1,num_params):
			product=product*param_lengths[j]*num_freqs  #### in model array, the last axis is that of frequency
			
		ind=ind+param_inds[i]*product
	return ind
	
cdef void calc_map_chi(double **spectrum, double *rms, double [:] sub_fitted,\
				double [:] model, int num_x, int num_y, int num_freqs,\
				int num_params, double sys_error):

	cdef int x1,y1,t,ind, num_times
	cdef double *spectrum_1d
	cdef double error
	cdef double sum1=0.0
	
	num_times=1
	
	t=0
	for y1 in range(num_y):
		for x1 in range(num_x):
			sum1=0.0
			ind=y1*num_x+x1
			spectrum_1d=spectrum[ind]
			ind=find_model_ind(sub_fitted,x1,y1,param_lengths,num_params, num_x, num_y)
			for i in range(num_freqs):
				if spectrum_1d[i]<0:
					continue
				error=sqrt(square(rms[i])+square(sys_error*spectrum_1d[i]))
				sum1=sum1+square((model[ind+i]-spectrum_1d[i])/error)
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
			sub_fitted[ind]=sum1
	return
	
cdef void get_random_param_ind(int *rand_param_inds, int num_params, int **param_val_ind, int *param_val_lengths):

	cdef double *seed
	seed=<double *>PyMem_Malloc(num_params*sizeof(double))
	
	gen_seed(seed, num_params)
	
	cdef int i
	cdef int rand_ind
	for i in range(num_params):
		rand_int=int(seed[i]*param_val_lengths[i])  ### seed[i] should lie between 0 and 1
		if rand_int<param_val_lengths[i]:
			rand_param_inds[i]=param_val_ind[i][rand_int]
		else:
			rand_param_inds[i]=0  #### fail-safe mechanism. This should not be triggered.
	
	PyMem_Free(seed)
	return
					

cdef void do_bee_swarm_optimisation(double **spectrum, double *rms, double [:] model,\
					int num_freqs, int **param_val_ind,int *param_val_lengths,\
					int num_params, int low_indx, int low_indy, int low_indt,\
					int high_indx, int high_indy, int high_indt, double [:] fitted,\
					double **param_val_actual, double sys_error, double spatial_smoothness_enforcer,\
					double temporal_smoothness_enforcer):
					
	cdef int x1,y1,i,ind,t,m
	
	cdef int num_val=(high_indx-low_indx+1)*(high_indy-low_indy+1)*(num_params+1)
	cdef double *subcube_fit
	cdef double *subcube_fit_copy
	cdef double *current_subcube_fit
	
	subcube_fit1=<double*>PyMem_Malloc(num_val*sizeof(double))
	
	subcube_fit_copy=<double*>PyMem_Malloc(num_val*sizeof(double))
	current_subcube_fit=<double*>PyMem_Malloc(num_val*sizeof(double))
	
	cdef double[:] sub_fitted=<double [:num_val]>subcube_fit
	
	
	i=0
	t=0
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx, high_indy+1):
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
			for m in range(num_params+1):
				sub_fitted[i]=fitted[ind+m]
				i=i+1
				
	
	for i in range(num_val):
		subcube_fit_copy[i]=sub_fitted[i]
		current_subcube_fit[i]=sub_fitted[i]
		
	cdef int num_y, num_x, num_times
	
	num_times=1
	num_x=high_indx-low_indx+1
	num_y=high_indy-low_indy+1
	
	
	cdef double *grad_x
	cdef double *grad_y
	cdef double *grad_t
	grad_x=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	grad_y=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	grad_t=<double *>PyMem_Malloc(num_times*num_y*num_x*num_params*sizeof(double))
	
	cdef double grad_chi=calc_grad_chi(sub_fitted, param_val, num_times, num_y, num_x, num_params, \
			grad_x, grad_y, grad_t, spatial_smoothness_enforcer, temporal_smoothness_enforcer)
	
	cdef double init_grad_chi=grad_chi
	cdef int max_iter=100
	cdef double exit_drop=0.2
	cdef double good_drop=0.1
	cdef int max_iter_good_drop=max_iter//2
	
	cdef int iter1=0
	cdef double drop=0.0
	cdef double current_grad_chi
	cdef double drop
	
	cdef int *rand_param_inds
	
	rand_param_inds=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	while iter1<max_iter:
		i=0
		t=0
		for y1 in range(low_indy,high_indy+1):
			for x1 in range(low_indx, high_indy+1):
				for m in range(num_params+1):
					sub_fitted[i]=-1.0
					i=i+1
		i=0
		t=0
		for y1 in range(low_indy,high_indy+1):
			for x1 in range(low_indx, high_indy+1):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
				get_random_param_ind(rand_param_inds,num_params,param_val_ind, param_val_lengths)
				for m in range(num_params):
					sub_fitted[i]=rand_param_inds[m]
					i=i+1
		calc_map_chi(spectrum, rms, sub_fitted,\
				model, num_x, num_y, num_freqs,\
				num_params, sys_error)
		current_grad_chi=calc_grad_chi(sub_fitted, param_val, num_times, num_y, num_x, num_params, \
			grad_x, grad_y, grad_t, spatial_smoothness_enforcer, temporal_smoothness_enforcer)
	
		if current_grad_chi<grad_chi:
			grad_chi=current_grad_chi
			for i in range(num_val):
				current_subcube_fit[i]=sub_fitted[i]
			drop=absolute((grad_chi-init_grad_chi)/grad_chi)
			if drop>exit_drop:
				break
			elif drop>good_drop and iter1>max_iter_good_drop:
				break
		iter1=iter1+1
	
	PyMem_Free(subcube_fit)
	PyMem_Free(grad_x)
	PyMem_Free(grad_y)
	PyMem_Free(grad_t)
	PyMem_Free(subcube_fit_copy)
	PyMem_Free(rand_param_inds)
	PyMem_Free(current_subcube_fit)
	return
	
		
	
cdef void make_data_ready_for_discont_removal(numpy.ndarray[numpy.double_t,ndim=4] cube,\
						numpy.ndarray[numpy.double_t,ndim=2]err_cube,\
						int *low_snr_loc, int ***low_freq_ind,\
						int ***upper_freq_ind, int num_times, int num_y,\
						int num_x, int num_freqs, int num_params,double **spectrum, double *rms,\
						int t0, int y0, int x0, int *low_indx, int *low_indy,\
						int *low_indt, int *high_indx, int *high_indy, int *high_indt,\
						int half_box_size, int **param_val_ind, double [:] fitted):
						
	cdef int low_indx_val, low_indy_val, low_indt_val
	cdef int high_indx_val,high_indy_val,high_indt_val
	
	
	low_indx_val=x0-half_box_size
	low_indy_val=y0-half_box_size
	
	high_indx_val=x0+half_box_size
	high_indy_val=y0+half_box_size
	
	high_indx_val=find_high_indx(high_indx_val,num_x)
	high_indy_val=find_high_indx(high_indy_val,num_y)
	low_indx_val=find_low_indx(low_indx_val)
	low_indy_val=find_low_indx(low_indy_val)
	
	low_indx[0]=low_indx_val
	low_indy[0]=low_indy_val
	low_indt[0]=0
	
	high_indx[0]=high_indx_val
	high_indy[0]=high_indy_val
	high_indt[0]=0	
	
	cdef int i,k,t
	cdef int t,j, ind, insert_position			
	
	
	
	for i in range((2*half_box_size+1)*(2*half_box_size+1)):
		for j in range(num_freqs):
			spectrum[i][j]=-1
	t=t0		
	for i in range(num_freqs):
		rms[i]=err_cube[t,i]
	
	t=t0
	i=0
	k=0		
	for y1 in range(low_indy_val,high_indy_val+1):
		for x1 in range(low_indx_val,high_indx_val+1):
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind]>0:
				for j in range(low_freq_ind[t][y1][x1],upper_freq_ind[t][y1][x1]+1):
					if low_snr_loc[t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs+j]==0:  ### not a low SNR point
						spectrum[i][j]=cube[t,j,y1,x1]
					
				for j in range(num_params):
					ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+j
					if fitted[ind]>0:
						insert_position=check_present_already(param_val_ind[j],int(fitted[ind]), (2*half_box_size+1)*(2*half_box_size+1))
						if insert_position>=0:
							param_val_ind[j][insert_position]=int(fitted[ind])
					
			i=i+1
	
				
	return
	
cdef void count_unique_param_vals(int **param_val_ind, int *param_val_lengths, int num_params, int box_size):
	cdef int i,j
	cdef int full_box=1
	for i in range(num_params):
		full_box=1
		for j in range(box_size):
			if param_val_ind[i][j]<0:
				param_val_lengths[i]=j
				full_box=0
				break	
		if full_box==1:
			param_val_lengths[i]=box_size
	return

cdef void remove_discontinuities(numpy.ndarray[numpy.double_t, ndim=4] cube,\
				numpy.ndarray[numpy.double_t,ndim=2] err_cube,\
				numpy.ndarray[numpy.double_t,ndim=1] model, \
				int *low_snr_loc, int ***low_freq_ind,\
				int ***upper_freq_ind,\
				double [:] fitted,double **param_val,\
				 int num_times, int num_x, int num_y, int num_params,\
				int num_freqs,int search_length, double thresh, \
				double **param_val_actual,double sys_error,\
				double spatial_smoothness_enforcer,\
				double temporal_smoothness_enforcer):
	
	cdef int tot_pix,i,j
					
	tot_pix=num_times*num_y*num_x  
	
	
	cdef int *discont[3]
	#cdef double 
	
	for i in range(3):
		discont[i]=<int *>PyMem_Malloc(tot_pix*sizeof(int))
		for j in range(tot_pix):
			discont[i][j]=-1
			
	cdef int discont_num=list_discont(fitted, param_val, num_x,num_y, num_times, num_params,\
			discont, search_length, thresh)
			
	cdef int box_size=2*search_length+1
	box_size=box_size*box_size
	cdef double **spectrum
	cdef double *rms
	cdef int **param_val_ind
	cdef int *param_val_lengths
	
	spectrum=<double **>PyMem_Malloc(box_size*sizeof(double *))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double *))
	param_val_ind=<int **>PyMem_Malloc(num_params*sizeof(int *))
	param_val_lengths=<int *>PyMem_Malloc(num_params*sizeof(int ))
	
	
	for i in range(box_size):
		spectrum[i]=<double *>PyMem_Malloc(num_freqs*sizeof(double))
		for j in range(num_freqs):
			spectrum[i][j]=-1.0
	for i in range(num_freqs):	
		rms[i]=-1.0
			
	for i in range(num_params):
		param_val_ind[i]=<int *>PyMem_Malloc(box_size*sizeof(int))
		param_val_lengths[i]=-1
		for  j in range(box_size):
			param_val_ind[i][j]=-1
	
	
	cdef int low_indx, low_indy, low_indt, high_indx, high_indy, high_indt
			
	for i in range(discont_num):
		make_data_ready_for_discont_removal(cube,err_cube,low_snr_loc,\
					low_freq_ind,upper_freq_ind,\
					num_times,num_y,num_x,num_freqs, \
					num_params,spectrum, rms, discont[0][i],\
					discont[1][i],discont[2][i], &low_indx,\
					&low_indy, &low_indt, &high_indx, &high_indy,\
					&high_indt,search_length, param_val_ind, fitted)
		
		count_unique_param_vals(param_val_ind,param_val_lengths, num_params, box_size)
		
		do_bee_swarm_optimisation(spectrum, rms, model,num_freqs, param_val_ind, param_val_lengths,\
					 num_params, low_indx,low_indy,low_indt,high_indx,high_indy,high_indt,\
					 fitted, param_val_actual, sys_error,spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
		
		print (i)	
		break
			
	for i in range(3):
		PyMem_Free(discont[i])
	
	for i in range(box_size):
		PyMem_Free(spectrum[i])
		
	for i in range(num_params):
		PyMem_Free(param_val_ind[i])
	
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(param_val_ind)
	PyMem_Free(param_val_lengths)
	return

cdef int detect_discont(double[:] fitted, double **param_val,int t0, int y, int x, int param,int num_x, \
			int num_y, int num_times, int num_params, int search_length,\
			 int axis, double thresh,int reverse):
	
	cdef int t,x1,y1,ind, ind1, ind5,j
	cdef double sum1, sum2, mean,std, all_mean, ratio
	
	sum1=0.0
	sum2=0.0
	
	ind5=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x*(num_params+1)+num_params
	if fitted[ind5]<-0.2:
		return 0
			
	cdef double param_value
	
	param_value=param_val[param][int(fitted[ind5-num_params+param])]
	
	
	cdef int num
	if axis==0:
		num=num_x
		ind=x
		
	elif axis==1:
		num=num_y
		ind=y
	else:
		num=num_times
		ind=t0
	
	ind1=ind
		
	cdef int low_ind, high_ind
	
	if reverse==0:
		low_ind=ind-search_length
		high_ind=ind
	elif reverse==1:
		low_ind=ind
		high_ind=ind+search_length
	
	if reverse==0:
		while low_ind<0 and high_ind<num-1:
			low_ind=low_ind+1
			high_ind=high_ind+1
	elif reverse==1:
		while high_ind>num-1 and low_ind>0:
			high_ind=high_ind-1
			low_ind=low_ind-1
		
	ind=high_ind-low_ind
	if ind<search_length:
		return 0
	
	if axis==0:
		j=0
		
		for x1 in range(low_ind,high_ind+1):
			if x1==ind1:
				continue
			ind5=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind5]>0.0:
				ind=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x1*(num_params+1)+param
				sum1=sum1+square(param_val[param][int(fitted[ind])])
				sum2=sum2+param_val[param][int(fitted[ind])]
				j=j+1
		
		if j<search_length//2:
			return 0
		mean=sum2/(search_length-1)
		std=sqrt(sum1/(search_length-1)-square(mean))
		all_mean=(sum2+param_value)/search_length
		if std<1e-4 and absolute(all_mean-mean)<1e-4:
			return 0
		elif std<1e-4 and absolute(all_mean-mean)>1e-4:
			return 1
		ratio=(absolute((all_mean-mean)/std))
		if ratio>thresh:
			return 1
	
	elif axis==1:
		j=0
		for y1 in range(low_ind,high_ind+1):
			if y1==ind1:
				continue
			ind5=t0*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x*(num_params+1)+num_params
			if fitted[ind5]>0.0:
				ind=t0*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x*(num_params+1)+param
				sum1=sum1+square(param_val[param][int(fitted[ind])])
				sum2=sum2+param_val[param][int(fitted[ind])]
				j=j+1
		if j<search_length//2:
			return 0
		mean=sum2/(search_length-1)
		std=sqrt(sum1/(search_length-1)-square(mean))
		all_mean=(sum2+param_value)/search_length
		if std<1e-4 and absolute(all_mean-mean)<1e-4:
			return 0
		elif std<1e-4 and absolute(all_mean-mean)>1e-4:
			return 1
		ratio=(absolute((all_mean-mean)/std))
		if ratio>thresh:
			return 1
	
	else:
		return 0  ### temporal discontinuity detection not implemented
	
	return 0



cdef int list_discont(double[:] fitted, double **param_val, int num_x, int num_y,\
			 int num_times, int num_params, int **discont, int search_length, double thresh):
	
			 
	cdef int t, x1,y1,param,j
	
	cdef int detected=0
	
	cdef int present_already=0
	
	#### remember that grad is a array with size num_times*num_y*num_x*num_params
	j=0
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				detected=0
				for param in range(num_params):
					detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
							 	num_params, search_length=search_length,thresh=thresh,axis=0,reverse=0)	
					if detected==1:
						break
						
					detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
							 	num_params, search_length=search_length,thresh=thresh,axis=1,reverse=0)
					if detected==1:
						break
					
					detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
							 	num_params, search_length=search_length,thresh=thresh,axis=2,reverse=0)	
					if detected==1:
						break
				if detected==0:
					for param in range(num_params):
						detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
								 	num_params, search_length=search_length,axis=0,thresh=thresh,reverse=1)	
						if detected==1:
							break
							
						detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
								 	num_params, search_length=search_length,axis=1,thresh=thresh,reverse=1)
						if detected==1:
							break
						
						detected=detect_discont(fitted,param_val,t,y1,x1,param,num_x,num_y,num_times,\
								 	num_params, search_length=search_length,axis=2,thresh=thresh,reverse=1)	
						if detected==1:
							break
				if detected==1:
					discont[0][j]=t
					discont[1][j]=y1
					discont[2][j]=x1
					j=j+1
					
			
	return j
	
	

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
	
	for i in range(num_freqs):
		smoothed_spectrum[i]=0.0
	
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
	
cdef void calc_grad(double [:]fitted,double **param_val, int num_times, int num_y, int num_x, int t, int y1, int x1, int num_params,\
					double * grad_x, double * grad_y, double * grad_t):


	cdef int ind,ind1, ind2,param1
	
	if num_x>1:
		if x1>=1 and x1<=num_x-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1-1)*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1+1)*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind1])])/2
				elif fitted[ind1-param1+num_params]>0:
					grad_x[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				elif fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_x[param1]=0.0
		elif x1==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1+1)*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_x[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_x[param1]=0.0
		elif x1==num_x-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+(x1-1)*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_x[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				else:
					grad_x[param1]=0.0
	if num_y>1:
		if y1>=1 and y1<=num_y-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+(y1-1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+(y1+1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind1])])/2
				elif fitted[ind1-param1+num_params]>0:
					grad_y[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				elif fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_y[param1]=0.0
		elif y1==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=t*num_y*num_x*(num_params+1)+(y1+1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_y[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_y[param1]=0.0
		elif y1==num_y-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=t*num_y*num_x*(num_params+1)+(y1-1)*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_y[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				else:
					grad_y[param1]=0.0
	
	if num_times>1:
		if t>=1 and t<=num_times-2:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=(t-1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=(t+1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0 and fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind1])])/2
				elif fitted[ind1-param1+num_params]>0:
					grad_t[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				elif fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_t[param1]=0.0
		elif t==0:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind2=(t+1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind2-param1+num_params]>0:
					grad_t[param1]=(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])])
				else:
					grad_t[param1]=0.0
		elif t==num_times-1:
			for param1 in range(num_params):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				ind1=(t-1)*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+param1
				if fitted[ind1-param1+num_params]>0:
					grad_t[param1]=(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])
				else:
					grad_t[param1]=0.0
	return

cdef double calc_grad_chi(double [:]fitted, double **param_val,int num_times, int num_y, int num_x, int num_params, \
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
					calc_grad(fitted,param_val,num_times,num_y,num_x,t,y1,x1,num_params,grad_x+ind3,grad_y+ind3,grad_t+ind3)
					
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
	
cdef void make_cube_fit_ready(int num_times,int num_y, int num_x,int num_freqs, double [:,:]err_cube,\
			double [:,:,:,:]cube, double *freqs1, double lower_freq,\
			double upper_freq, int ***low_freq_ind, int ***upper_freq_ind, int min_freq_num,\
			int num_params, double [:] fitted, int *pos, double rms_thresh):

	cdef int t, i, y1, x1,j, low_ind, high_ind, low_snr_freq_num,l, ind5, ind3
	cdef double *spectrum
	cdef double *rms
	
	spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	
	for t in range(num_times):
		for i in range(num_freqs):
			rms[i]=err_cube[t,i]
		for y1 in range(num_y):
			for x1 in range(num_x):
				for i in range(num_freqs):
					spectrum[i]=cube[t,i,y1,x1]
				ind3=t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs		
				low_ind=find_min_freq(freqs1,lower_freq,num_freqs)
				high_ind=find_max_freq(freqs1,upper_freq,num_freqs)
				calc_fitrange_homogenous(spectrum, &low_ind, &high_ind,num_freqs)
				low_freq_ind[t][y1][x1]=low_ind
				upper_freq_ind[t][y1][x1]=high_ind
				low_snr_freq_num=detect_low_snr_freqs(spectrum,rms,rms_thresh,pos+ind3,num_freqs)
				if low_snr_freq_num>min_freq_num:
					for l in range(num_params):
						ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l
						fitted[ind5]=-1
					ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
					fitted[ind5]=-1
				else:
					ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
					fitted[ind5]=0.00
					
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	return
	
cdef double calc_red_chi_all_pix(int num_times, int num_freqs, int num_y, int num_x, int num_params,int ***low_freq_ind,\
						int ***upper_freq_ind, numpy.ndarray[numpy.double_t,ndim=4] cube,\
						numpy.ndarray[numpy.double_t,ndim=2] err_cube,numpy.ndarray[numpy.double_t,ndim=1] model, double sys_error,\
						int *pos,double [:] fitted, double *freqs1, double lower_freq,\
						double upper_freq, int first_pass, double rms_thresh, int min_freq_num, int *param_lengths1,\
						int *param_ind, double **param_val,double spatial_smoothness_enforcer,\
						double temporal_smoothness_enforcer, int search_length, double discont_thresh):
						
						
	cdef double *spectrum
	cdef double *rms
	cdef double *sys_err
	cdef double *error
	cdef double red_chi
	cdef int low_ind,high_ind, t, x1,y1,j, ind3,ind
	cdef int low_snr_freq_num
	cdef int ind5
	'''
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
					error[j]=sqrt(square(rms[j])+square(sys_err[j]))
				
				ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				if fitted[ind5]<-0.2:
					continue
				ind3=t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs	
				red_chi=min_chi_square(model,spectrum,error,low_freq_ind[t][y1][x1],upper_freq_ind[t][y1][x1],\
							rms_thresh,rms,num_params,num_freqs,param_lengths1,pos+ind3,sys_error,param_ind)
				
				for l in range(num_params):
					ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+l
					fitted[ind5]=param_ind[l]#param_val[l][int(param_ind[l])]
				ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				fitted[ind5]=red_chi	
					
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(sys_err)
	PyMem_Free(error)
	
	###------------------------------------------ For testing the code after initial value finding---------------------####
	
	hf=h5py.File("test_cython_code.hdf5","w")
	hf.create_dataset("fitted",data=fitted)
	hf.close()
	'''
	hf=h5py.File('test_cython_code.hdf5')
	cdef numpy.ndarray[numpy.double_t,ndim=1]fitted1
	fitted1=np.array(hf['fitted'])
	hf.close()
	
	####------------------------------------ for testing---------------------------------####
	
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
					
	cdef double grad_chi=calc_grad_chi(fitted1,param_val,num_times,num_y,num_x,num_params, \
						grad_x,grad_y,grad_t, spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
						
	
				
	remove_discontinuities(cube, err_cube, model, pos, low_freq_ind, upper_freq_ind, fitted1,param_val,num_times,num_x, num_y, \
				num_params,num_freqs,search_length,discont_thresh, param_val, sys_error,spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
	
						
	
	return grad_chi
	

cdef double absolute (double x):
	if x<0:
		return -x
	return x				
cpdef numpy.ndarray[numpy.double_t,ndim=1] compute_min_chi_square(numpy.ndarray[numpy.double_t, ndim=1] model, \
				numpy.ndarray[numpy.double_t,ndim=4] cube,\
				numpy.ndarray[numpy.double_t,ndim=2] err_cube,\
				double lower_freq,\
				double upper_freq,\
				numpy.ndarray[numpy.int_t,ndim=1] param_lengths,\
				numpy.ndarray[numpy.double_t,ndim=1] freqs,
				double sys_error, double rms_thresh, int min_freq_num, int num_params,\
				int num_times, int num_freqs, int num_y,int num_x, numpy.ndarray[numpy.double_t,ndim=1]param_vals,\
				double spatial_smoothness_enforcer=0.001,double temporal_smoothness_enforcer=0.0,\
				double frac_tol=0.1, int max_iter=10, int search_length=20, double discont_thresh=3.0):
				
	cdef int t,y1,x1,i,j,l,ind
	cdef numpy.ndarray[numpy.double_t,ndim=1] fitted1=np.zeros(num_times*num_y*num_x*(num_params+1))
	
	cdef int *pos
	pos=<int *>PyMem_Malloc(num_times*num_y*num_x*num_freqs*sizeof(int))
	
	for i in range(num_times*num_y*num_x*num_freqs):
		pos[i]=-1
	
	cdef int *param_ind
	param_ind=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	for i in range(num_params):
		param_ind[i]=-1
	
	
	cdef int low_snr_freq_num
	cdef double red_chi
	
	cdef int *param_lengths1
	param_lengths1=<int *>PyMem_Malloc(num_params*sizeof(int))
	for i in range(num_params):
		param_lengths1[i]=param_lengths[i]  #### making it a raw pointer. Faster access
	
	cdef double *freqs1
	freqs1=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	for i in range(num_freqs):
		freqs1[i]=freqs[i]
	
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
			for x1 in range(num_x):
				low_freq_ind[t][y1][x1]=0
				upper_freq_ind[t][y1][x1]=0
	
	
	
	
	cdef int first_try=0	
	
	
	
	make_cube_fit_ready(num_times, num_y, num_x,num_freqs, err_cube,\
			cube, freqs1, lower_freq,\
			upper_freq, low_freq_ind, upper_freq_ind,min_freq_num,\
			num_params, fitted1, pos,rms_thresh)
			
			
	cdef double **param_vals1
	param_vals1=<double **>PyMem_Malloc(num_params*sizeof(double **))
	
	
	l=0
	for i in range(num_params):
		param_vals1[i]=<double *>PyMem_Malloc(param_lengths1[i]*sizeof(double))
		for j in range(param_lengths1[i]):
			param_vals1[i][j]=param_vals[l]
			l=l+1
			
			
			
				
	cdef double grad_chi=calc_red_chi_all_pix(num_times, num_freqs, num_y, num_x, num_params,low_freq_ind,\
						upper_freq_ind, cube, err_cube, model, sys_error,\
						pos, fitted1, freqs1, lower_freq,\
						upper_freq, first_try, rms_thresh, min_freq_num,\
						param_lengths1,param_ind, param_vals1,\
						spatial_smoothness_enforcer, temporal_smoothness_enforcer,\
						search_length=search_length,discont_thresh=discont_thresh)
						
						
	print (grad_chi)	
	
				
	
	for t in range(num_times):
		for y1 in range(num_y):
			PyMem_Free(low_freq_ind[t][y1])
			PyMem_Free(upper_freq_ind[t][y1])
		PyMem_Free(low_freq_ind[t])
		PyMem_Free(upper_freq_ind[t])	
		
		
	for i in range(num_params):
		PyMem_Free(param_vals1[i])
	PyMem_Free(param_vals1)
	PyMem_Free(low_freq_ind)
	PyMem_Free(upper_freq_ind)	
			
	PyMem_Free(pos)
	PyMem_Free(param_ind)
	return fitted1
				
