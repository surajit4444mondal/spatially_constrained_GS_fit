cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython	

cdef int find_min_freq(double * freqs,\
			double lower_freq,\
			int num_freqs):
	cdef int i
	if freqs[0]>=lower_freq:
		return 0
	for i in range(1,num_freqs):
		if freqs[i]==lower_freq:
			return i
		if freqs[i]>lower_freq and freqs[i-1]<lower_freq:
			return i

cdef int find_max_freq(double * freqs,\
			double upper_freq,\
			int num_freqs):
	cdef int i
	for i in range(num_freqs-1):
		if freqs[i]==upper_freq:
			return i
		if freqs[i]<upper_freq and freqs[i+1]>upper_freq:
			return i
	if freqs[num_freqs-1]<=upper_freq:
		return num_freqs-1		

cdef int detect_low_snr_freqs(double *spectrum,\
				double *rms,\
				 double rms_thresh,\
				  int *pos,\
				   int low_ind,\
				    int high_ind,\
				    int num_freqs):
	'''
	counts the number of high SNR freqs
	'''
	cdef int i
	cdef int j
	j=0
	for i in range(num_freqs):
		if rms[i]<1e-5:
			pos[i]=1
			
		elif spectrum[i]<rms_thresh*rms[i]:
			pos[i]=1
		else:
			pos[i]=0
			if i>=low_ind and i<=high_ind:
				j=j+1
	return j

cdef double square(double x):
	return x*x	
	

cdef double min_chi_square(double *model,\
			   double *spectrum,\
			   double *error,\
			   int low_ind,\
			   int high_ind, \
			   double rms_thresh,\
			    double *rms, \
			    int num_params,\
			     int num_freqs, \
			     int *param_lengths,\
			      int *low_snr_freqs,\
		             double sys_error,\
		              int * param_inds):
	
	cdef double min_chi=1e100
	cdef unsigned int i,j,min_ind
	cdef double chi=0.0
	cdef int k=0
	cdef int mid_ind
	cdef double ratio
	
	cdef unsigned int num_elem_model,num_param_comb=1
	
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
	
cdef int find_max_pos(double *spectrum,\
			 int *lower_freq_ind,\
			  int *upper_freq_ind,\
			   int num_freqs):
	cdef int i, max_loc
	cdef double max_val=-1.0
	for i in range(lower_freq_ind[0],upper_freq_ind[0]+1):
		if spectrum[i]>max_val:
			max_loc=i
			max_val=spectrum[i]
	return max_loc
			
		
cdef void calc_fitrange_homogenous(double *spectrum, \
					int *lower_freq_ind,\
					 int *upper_freq_ind,\
					 int num_freqs,\
					  double *error,\
					   double sys_error):
					   
	cdef int max_loc=find_max_pos(spectrum,lower_freq_ind, upper_freq_ind, num_freqs)
			
	cdef double *smoothed_spectrum
	smoothed_spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	
		
	cdef int smooth_length=3
	cdef int i,j
	cdef int smooth_interval=smooth_length//2
	cdef double sum1
	
	for i in range(num_freqs):
		smoothed_spectrum[i]=0.0
	
	for i in range(lower_freq_ind[0],upper_freq_ind[0]+1):
		if i<smooth_interval:
			smoothed_spectrum[i]=0.0
			continue
		elif i>num_freqs-1-smooth_interval:
			smoothed_spectrum[i]=0.0
			continue
		j=-smooth_interval
		sum1=0.0
		while j<smooth_interval:
			sum1=sum1+spectrum[i+j]
			j=j+1
		sum1=sum1/smooth_length
		smoothed_spectrum[i]=sum1
	
	
	j=max_loc-smooth_interval
	
	
	while j>smooth_interval:
		if smoothed_spectrum[j]<1e-7:
			lower_freq_ind[0]=j
			break
		if j==lower_freq_ind[0]:
			break
		if (smoothed_spectrum[j-1]-smoothed_spectrum[j])>sqrt((square(error[j])+\
								   square(error[j-1]))/smooth_length+\
								   square(sys_error*smoothed_spectrum[j-1])+\
								   square(sys_error*smoothed_spectrum[j])):
			lower_freq_ind[0]=j
			break
		j=j-1
		
	j=max_loc+1+smooth_interval
	
	
	while j<num_freqs-2-smooth_interval:
		if smoothed_spectrum[j]<1e-7:
			upper_freq_ind[0]=j
			break
		if j==upper_freq_ind[0]:
			break
		if (smoothed_spectrum[j+1]-smoothed_spectrum[j])>sqrt((square(error[j])+\
								   square(error[j-1]))/smooth_length+\
								   square(sys_error*smoothed_spectrum[j+1])+\
								   square(sys_error*smoothed_spectrum[j])):
			upper_freq_ind[0]=j
			break
		j=j+1
		
	PyMem_Free(smoothed_spectrum)
	return
	
cdef int find_maximum(int *param_lengths,\
			int num_params):
	
	cdef int i, max1
	
	max1=0
	
	for i in range(num_params):
		if max1<param_lengths[i]:
			max1=param_lengths[i]
	return max1
	
cdef double calc_gradx(int x0,\
		int y0,\
		int param,\
		double *fitted,\
		int numx,\
		int numy,\
		int num_params,\
		double smoothness_enforcer):
	
	
	cdef int x1=x0-1
	cdef int y1=y0
	cdef int ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	cdef int ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	cdef double grad1=0
	if x1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		grad1=(fitted[ind]-fitted[ind1])
		
	cdef int x2=x0+1
	cdef int y2=y0
	cdef int ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	cdef double grad2=0
	
	if x2>numx-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		grad2=(fitted[ind2]-fitted[ind])
		
	cdef double grad=sqrt(square(grad1)+square(grad2))
	return grad
	
cdef double calc_grady(int x0,\
		int y0,\
		int param,\
		double *fitted,\
		int numx,\
		int numy,\
		int num_params,\
		double smoothness_enforcer):
	
	cdef int x1=x0
	cdef int y1=y0-1
	cdef int ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	cdef int ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	cdef double grad1=0
	if y1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		grad1=(fitted[ind]-fitted[ind1])
		
	cdef int x2=x0
	cdef int y2=y0+1
	cdef int ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	cdef double grad2=0
	
	if y2>numy-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		grad2=(fitted[ind2]-fitted[ind])
		
	cdef double grad=sqrt(square(grad1)+square(grad2))
	return grad	
	
cdef double calc_gradient(int x0,\
			   int y0,\
			   double *fitted,\
			   int num_x,\
			   int num_y,\
			   int num_params,\
			   int *param_lengths,\
			   double smoothness_enforcer):
			   
	cdef int ind,param
	ind=y0*num_x*(num_params+1)+x0*(num_params+1)+num_params
	if fitted[ind]<0:
		return 0

	cdef double grad=0.0	
		
	cdef int max_param_length=find_maximum(param_lengths,num_params)
		
	cdef double gradx,grady
	for param in range(num_params):
		gradx=calc_gradx(x0,y0,param,fitted, num_x,num_y,num_params,smoothness_enforcer)
		grady=calc_grady(x0,y0,param,fitted, num_x,num_y,num_params,smoothness_enforcer)
		grad=grad+(square(gradx)+square(grady))*smoothness_enforcer*max_param_length/param_lengths[param]
	return grad	
	
cpdef double calc_chi_square(double [:]spectrum,\
				 double [:]rms,\
				 double sys_error,\
				 double[:] model_spectrum,\
				 int low_ind,\
				 int high_ind,\
				 double rms_thresh):
	cdef double chi_square=0
	cdef int freq_ind
	cdef double error
	for freq_ind in range(low_ind,high_ind+1):
		if spectrum[freq_ind]>=rms_thresh*rms[freq_ind]:
			error=sqrt(square(rms[freq_ind])+square(sys_error*spectrum[freq_ind]))
			chi_square+=square(spectrum[freq_ind]-model_spectrum[freq_ind])/square(error)
		else:
			if model_spectrum[freq_ind]<(1+sys_error)*rms_thresh*rms[freq_ind]:
				chi_square+=0.0
			else:
				chi_square+=1000.0
	return chi_square
	
cpdef double calc_grad_chisquare(int low_indx,\
				  int low_indy,\
				  int high_indx,\
				  int high_indy,\
				  int numx,\
				  int numy, \
				  int num_params,\
				  double [:] fitted,\
				  int [:] param_lengths,\
				  double smoothness_enforcer):
				  
	cdef double chi_square=0.0
	cdef int y1,x1,ind
	
	cdef double *fitted1
	fitted1=&fitted[0] 
	
	cdef int *param_lengths1
	param_lengths1=&param_lengths[0]
	
	for y1 in range(low_indy, high_indy):
		for x1 in range(low_indx,high_indx):
			ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind]>0:
				chi_square=chi_square+fitted[ind]+\
						calc_gradient(x1,y1,fitted1,\
						numx,numy,num_params,param_lengths1,\
						smoothness_enforcer)
						
	return chi_square
	
cdef void make_cube_fit_ready(int num_times,\
				int num_y,\
				 int num_x,\
				 int num_freqs,\
				  double *err_cube,\
				double *cube,\
				double *freqs1,\
				 double lower_freq,\
				double upper_freq,\
				int *low_freq_ind,\
				int *upper_freq_ind,\
				 int min_freq_num,\
				int num_params,\
				 double * fitted, \
				 int *pos,\
				 double rms_thresh,\
				 double sys_error):

	cdef int t, i, y1, x1,j, low_ind, high_ind, high_snr_freq_num,l, ind5, ind3
	cdef double *spectrum
	cdef double *rms
	cdef int freq_ind
	
	spectrum=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	rms=<double *>PyMem_Malloc(num_freqs*sizeof(double))
	
	
	for t in range(num_times):
		for i in range(num_freqs):
			freq_ind=t*num_freqs+i
			rms[i]=err_cube[freq_ind]
		for y1 in range(num_y):
			for x1 in range(num_x):
				for i in range(num_freqs):
					freq_ind=t*num_freqs*num_y*num_x+i*num_y*num_x+y1*num_x+x1
					spectrum[i]=cube[freq_ind]
				ind3=t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs		
				low_ind=find_min_freq(freqs1,lower_freq,num_freqs)
				high_ind=find_max_freq(freqs1,upper_freq,num_freqs)
				freq_ind=t*num_freqs
				calc_fitrange_homogenous(spectrum, &low_ind, &high_ind,num_freqs,err_cube+freq_ind, sys_error)
				
				freq_ind=t*num_y*num_x+y1*num_x+x1
				low_freq_ind[freq_ind]=low_ind
				upper_freq_ind[freq_ind]=high_ind
				high_snr_freq_num=detect_low_snr_freqs(spectrum,rms,rms_thresh,pos+ind3,low_ind,high_ind,num_freqs)
				
				if high_snr_freq_num<min_freq_num:
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
	
cdef void calc_red_chi_all_pix(int num_times,\
				 int num_freqs,\
				  int num_y,\
				   int num_x, \
				   int num_params,\
				   int *low_freq_ind,\
				   int *upper_freq_ind, \
				   double *cube,\
				  double * err_cube,\
				  double *model, \
				  double sys_error,\
				  int *pos,\
				  double *fitted, \
				  double *freqs1, \
				  double lower_freq,\
				 double upper_freq,\
				  double rms_thresh,\
				   int min_freq_num, \
				   int *param_lengths1,\
				   int *param_ind):
						
	
				
	cdef double *spectrum
	cdef double *rms
	cdef double *sys_err
	cdef double *error
	cdef double red_chi
	cdef int low_ind,high_ind, t, x1,y1,j, ind3,ind,freq_ind
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
			freq_ind=t*num_freqs+i
			rms[i]=err_cube[freq_ind]
		for y1 in range(num_y):
			for x1 in range(num_x):
				for j in range(num_freqs):
					freq_ind=t*num_freqs*num_y*num_x+j*num_y*num_x+y1*num_x+x1
					spectrum[j]=cube[freq_ind]
					sys_err[j]=sys_error*spectrum[j]
					error[j]=sqrt(square(rms[j])+square(sys_err[j]))
				
				ind5=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				if fitted[ind5]<-0.2:
					continue
				ind3=t*num_y*num_x*num_freqs+y1*num_x*num_freqs+x1*num_freqs	
				freq_ind=t*num_y*num_x+y1*num_x+x1
				red_chi=min_chi_square(model,spectrum,error,low_freq_ind[freq_ind],upper_freq_ind[freq_ind],\
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
	
		
	return
	

cdef double absolute (double x):
	if x<0:
		return -x
	return x				
cpdef void compute_min_chi_square(double[::1] model, \
				double[::1]cube,\
				double [::1] err_cube,\
				double lower_freq,\
				double upper_freq,\
				int[::1] param_lengths,\
				double[::1] freqs,
				double sys_error, \
				double rms_thresh, \
				int min_freq_num, \
				int num_params,\
				int num_times,\
				 int num_freqs, \
				 int num_y,\
				 int num_x, \
				 double [::1]param_vals,\
				 int [:] high_snr_freq_loc,\
				 double [:] fitted,\
				 int [:] low_freq_ind,\
				 int [:] upper_freq_ind):
				
	cdef int t,y1,x1,i,j,l,ind
	cdef double *fitted1
	cdef double *model1
	cdef double *cube1
	cdef double *err_cube1
	fitted1=&fitted[0]
	model1=&model[0]
	cube1=&cube[0]
	err_cube1=&err_cube[0]
	cdef int *pos
	pos=&high_snr_freq_loc[0]
	
	for i in range(num_times*num_y*num_x*num_freqs):
		pos[i]=-1
	
	cdef int *param_ind
	param_ind=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	for i in range(num_params):
		param_ind[i]=-1
	
	
	cdef int low_snr_freq_num
	cdef double red_chi
	
	cdef int *param_lengths1
	param_lengths1=&param_lengths[0]
	
	
	cdef double *freqs1
	freqs1=&freqs[0]
	
	
	cdef int *low_freq_ind1
	cdef int *upper_freq_ind1
	
	low_freq_ind1=&low_freq_ind[0]
	upper_freq_ind1=&upper_freq_ind[0]
	
	make_cube_fit_ready(num_times, num_y, num_x,num_freqs, err_cube1,\
			cube1, freqs1, lower_freq,\
			upper_freq, low_freq_ind1, upper_freq_ind1,min_freq_num,\
			num_params, fitted1, pos,rms_thresh,sys_error)
					
	calc_red_chi_all_pix(num_times, num_freqs, num_y, num_x, num_params,low_freq_ind1,\
						upper_freq_ind1, cube1, err_cube1, model1, sys_error,\
						pos, fitted1, freqs1, lower_freq,\
						upper_freq,rms_thresh, min_freq_num,\
						param_lengths1,param_ind)
									
	PyMem_Free(param_ind)
	return 
				
