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

cdef struct fof_struct:
	int x
	int y
	int t
	double *fof_params
	fof_struct *cluster_forward
	fof_struct *cluster_backward
	fof_struct *member_forward
	fof_struct *member_backward
	
cdef void make_fof_struct(int x0, int y0, int t0, int num_times, int num_y, \
			int num_x, int num_params, double [:] fitted, double **param_vals,\
			 fof_struct *fof):
	cdef int i
	cdef unsigned int ind
	
	ind=t0*num_y*num_x*(num_params+1)+y0*num_x*(num_params+1)+x0*(num_params+1)
	
	fof.cluster_forward=NULL
	fof.cluster_backward=NULL
	fof.member_forward=NULL
	fof.member_backward=NULL
	cdef double *fof_params
	fof_params=<double *>PyMem_Malloc(num_params*sizeof(double))
	
	for i in range(num_params):
		fof_params[i]=param_vals[i][int(fitted[ind+i])]
	
	fof.fof_params=fof_params
	
	fof.x=x0
	fof.y=y0
	fof.t=t0
	return
	
	
cdef void add_first_cluster(fof_struct *fof_list, fof_struct *cluster):
	fof_list=cluster
	return
	
cdef void add_clusters(fof_struct *fof_list, fof_struct *cluster):
	cdef fof_struct *temp
	temp=fof_list
	
	while temp.cluster_forward!=NULL:
		temp=temp.cluster_forward
	### when the while loop exits, temp is a pointer to the last element of the cluster list
	
	temp.cluster_forward=cluster
	cluster.cluster_backward=temp
	
	return
	
cdef void add_member(fof_struct *fof_list, int cluster_number,fof_struct *member):
	cdef int i
	i=0
	cdef fof_struct *temp
	temp=fof_list
	
	while i<cluster_number:
		temp=temp.cluster_forward
		i=i+1
		
	while temp.member_forward!=NULL:
		temp=temp.member_forward

	temp.member_forward=member
	member.member_backward=temp
	return		
	
cdef void join_clusters(fof_struct *fof_list, int cluster_number1, int cluster_number2):
	cdef int i
	
	cdef fof_struct *temp1
	temp1=fof_list
	
	i=0
	while i<cluster_number1:
		temp1=temp1.cluster_forward
		i=i+1
		
	cdef fof_struct *temp2
	temp2=fof_list
	
	i=0
	temp2=fof_list
	while temp2.cluster_forward!=NULL and i<cluster_number2:
		temp2=temp2.cluster_forward
		i=i+1
	cdef fof_struct *temp3


	temp3=temp2.cluster_backward
	
	temp3.cluster_forward=temp2.cluster_forward
	if temp2.cluster_forward!=NULL:
		temp2.cluster_forward.cluster_backward=temp3
	while temp1.member_forward!=NULL:
		temp1=temp1.member_forward
	
	
	temp1.member_forward=temp2
	temp2.member_backward=temp1	
	temp2.cluster_forward=NULL
	temp2.cluster_backward=NULL
	return
	
cdef free_members(fof_struct *member_list):
	cdef fof_struct *temp2
	
	temp2=member_list
	
	while temp2.member_forward!=NULL:
		PyMem_Free(temp2.fof_params)
		temp2=temp2.member_forward
		
	while temp2.member_backward!=NULL:
		temp2=temp2.member_backward
		
		PyMem_Free(temp2.member_forward)
		
	return
	
	
cdef void free_cluster(fof_struct *fof_list):
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	
	temp1=fof_list
	
	while temp1.cluster_forward!=NULL:
		temp2=temp1
		free_members(temp2)
		temp1=temp1.cluster_forward
	
	free_members(temp1)
	temp1=fof_list
	
	while temp1.cluster_forward!=NULL:
		temp1=temp1.cluster_forward
		
	while temp1.cluster_backward!=NULL:
		temp1=temp1.cluster_backward
		PyMem_Free(temp1.cluster_forward)
	
	PyMem_Free(fof_list)
	return

	
cdef void cluster_to_add(fof_struct *fof_list, fof_struct *member_struct, double max_dist, int *cluster_numbers, int num_params):
	
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	cdef double dist
	cdef int i,j
	
	temp1=fof_list
	i=0
	j=0
	while temp1!=NULL:
		temp2=temp1
		while temp2!=NULL:
			dist=find_distance(temp2,member_struct, num_params)
			if dist<=max_dist:
				cluster_numbers[j]=i
				j=j+1
				if j==2:
					return
				break
			temp2=temp2.member_forward
		temp1=temp1.cluster_forward
		i=i+1
	return
cdef void find_member_number(fof_struct *fof_list, int *member_number):
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	cdef double dist
	cdef int i,j
	
	temp1=fof_list
	i=0
	j=0
	while temp1!=NULL:
		temp2=temp1
		j=0
		while temp2!=NULL:
			j=j+1
			temp2=temp2.member_forward
		temp1=temp1.cluster_forward
		member_number[i]=j
		i=i+1
	return

cdef int print_cluster(fof_struct *fof_list):
	
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	cdef double dist
	cdef int i,j
	
	temp1=fof_list
	i=0
	j=0
	while temp1!=NULL:
		print ("new_cluster")
		temp2=temp1
		while temp2!=NULL:
			print (temp2.x,temp2.y)
			temp2=temp2.member_forward
		temp1=temp1.cluster_forward
		i=i+1
	return i
	
cdef int add_locs_to_change(int cluster_num, int **locations_to_change, fof_struct *fof_list, int max_size):
	cdef int i,inserted
	inserted=0
	for i in range(max_size):
		if locations_to_change[0][i]<0:
			inserted=1
			break
	if inserted==0:
		return inserted
	
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	cdef double dist
	cdef int j
	
	temp1=fof_list
	j=0
	while temp1!=NULL:
		temp2=temp1
		if j==cluster_num:
			while temp2!=NULL:
				locations_to_change[0][i]=temp2.x
				locations_to_change[1][i]=temp2.y
				i=i+1
				temp2=temp2.member_forward
			break
		temp1=temp1.cluster_forward
		j=j+1
	return inserted	
	
cdef int find_cluster_num(fof_struct *fof_list):
	
	cdef fof_struct *temp1
	cdef fof_struct *temp2
	cdef double dist
	cdef int i,j
	
	temp1=fof_list
	i=0
	j=0
	while temp1!=NULL:
		temp1=temp1.cluster_forward
		i=i+1
	return i

cdef int find_high_indx(int high_ind_val, int num):
	cdef int i
	i=high_ind_val
	while i>=num:
		i=i-1
	return i

cdef int find_low_indx(int low_ind_val):
	cdef int i=low_ind_val
	while i<0:
		i=i+1
	return i
	
cdef int check_present_already(int **param_val_ind,int *current_param_ind, int max_size, int num_params):
	cdef int i, j
	cdef int present=1
	for i in range(max_size):
		present=1
		for j in range(num_params):
			if param_val_ind[j][i]<0:
				return i
			
			if param_val_ind[j][i]!=current_param_ind[j]:
				present=0
				break
		if present==1:
			return -1
		
	
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
	
cdef unsigned int find_model_ind(double [:] fitted, int x, int y, int *param_lengths, int num_params,\
			int num_x, int num_y, int num_freqs):
	
	cdef unsigned int i,ind,j, product
	ind=y*num_x*(num_params+1)+x*(num_params+1)
	
	
	cdef int *param_inds
	param_inds=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	for i in range(num_params):
		param_inds[i]=int(fitted[ind+i])
	
	ind=0
	for i in range(num_params):
		product=1
		for j in range(i+1,num_params):
			product=product*param_lengths[j]
		ind=ind+param_inds[i]*product*num_freqs  #### in model array, the last axis is that of frequency
	return ind
	
cdef void calc_map_chi(double **spectrum, double *rms, double [:] sub_fitted,\
				double [:] model, int num_x, int num_y, int num_freqs,\
				int num_params, double sys_error, int *param_lengths):

	cdef int x1,y1,t, num_times
	cdef double *spectrum_1d
	cdef double error
	cdef double sum1=0.0
	cdef unsigned int ind
	
	num_times=1
	
	t=0
	cdef int k
	for y1 in range(num_y):
		for x1 in range(num_x):
			sum1=0.0
			ind=y1*num_x+x1
			spectrum_1d=spectrum[ind]
			ind=find_model_ind(sub_fitted,x1,y1,param_lengths,num_params, num_x, num_y, num_freqs)  
			k=0
			for i in range(num_freqs):
				if spectrum_1d[i]<0:
					continue
				error=sqrt(square(rms[i])+square(sys_error*spectrum_1d[i]))
				sum1=sum1+square((model[ind+i]-spectrum_1d[i])/error)
				k=k+1
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
			if k>0:
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
					double temporal_smoothness_enforcer, int num_x, int num_y, int *model_param_lengths):
		
		
				
	cdef int x1,y1,i,ind,t,m
	
	cdef int num_val=(high_indx-low_indx+1)*(high_indy-low_indy+1)*(num_params+1)
	cdef double *subcube_fit
	cdef double *subcube_fit_copy
	cdef double *current_subcube_fit
	
	subcube_fit=<double*>PyMem_Malloc(num_val*sizeof(double))
	current_subcube_fit=<double*>PyMem_Malloc(num_val*sizeof(double))
	
	cdef double[:] sub_fitted=<double [:num_val]>subcube_fit
	
	i=0
	t=0
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx, high_indx+1):
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
			for m in range(num_params+1):
				sub_fitted[i]=fitted[ind+m]
				i=i+1
			
				
			
	
	for i in range(num_val):
		current_subcube_fit[i]=sub_fitted[i]
		
	cdef int num_y1, num_x1, num_times
	
	
	num_times=1
	num_x1=high_indx-low_indx+1
	num_y1=high_indy-low_indy+1
	
	
	
	
	cdef double grad_chi=calc_grad_chi(sub_fitted, param_val_actual, num_times, num_y1, num_x1, num_params, \
					 spatial_smoothness_enforcer, temporal_smoothness_enforcer)
	
	
	cdef double init_grad_chi=grad_chi
	cdef int max_iter=1000
	cdef double exit_drop=0.6
	cdef double good_drop=0.4
	cdef int max_iter_good_drop=max_iter//2
	
	cdef int iter1=0
	cdef double drop=0.0
	cdef double current_grad_chi
	
	
	cdef int *rand_param_inds
	
	rand_param_inds=<int *>PyMem_Malloc(num_params*sizeof(int))
	
	while iter1<max_iter:
		i=0
		t=0
		for y1 in range(low_indy,high_indy+1):
			for x1 in range(low_indx, high_indx+1):
				for m in range(num_params+1):
					sub_fitted[i]=-1.0
					i=i+1
		
		i=0
		t=0
		for y1 in range(num_y1):
			for x1 in range(num_x1):
				ind=t*num_y1*num_x1*(num_params+1)+y1*num_x1*(num_params+1)+x1*(num_params+1)
				get_random_param_ind(rand_param_inds,num_params,param_val_ind, param_val_lengths)
				for m in range(num_params):
					sub_fitted[ind+m]=rand_param_inds[m]
				
		calc_map_chi(spectrum, rms, sub_fitted,\
				model, num_x1, num_y1, num_freqs,\
				num_params, sys_error,model_param_lengths)
				
		current_grad_chi=calc_grad_chi(sub_fitted, param_val_actual, num_times, num_y1, num_x1, num_params, \
						 spatial_smoothness_enforcer, temporal_smoothness_enforcer)
		
		if current_grad_chi<grad_chi:
			grad_chi=current_grad_chi
			print ("successfull")
			for i in range(num_val):
				current_subcube_fit[i]=sub_fitted[i]
			drop=absolute((grad_chi-init_grad_chi)/init_grad_chi)
			if drop>exit_drop:
				break
			elif drop>good_drop and iter1>max_iter_good_drop:
				break
		
		iter1=iter1+1
	
	i=0
	t=0	
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx, high_indx+1):
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
			for m in range(num_params+1):
				fitted[ind+m]=current_subcube_fit[i]
				i=i+1
	
	PyMem_Free(subcube_fit)
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
						int half_box_size, int **param_val_ind, double [:] fitted,\
						double *previous_param_inds):
						
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
	
	
	cdef int i,k
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
					
			i=i+1
	
	t=t0
	i=0
	k=0
	for y1 in range(low_indy_val,high_indy_val+1):
		for x1 in range(low_indx_val,high_indx_val+1):
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
			for j in range(num_params+1):
				previous_param_inds[i]=fitted[ind+j]
				i=i+1
			
	return
	
cdef get_high_confidence_param_vals(int low_indx, int low_indy, int low_indt, int high_indx, int high_indy, int high_indt,\
					 int num_times, int num_y, int num_x, int num_params,double[:] fitted, int **param_val_ind,\
					  int **locations_to_change, int max_locations_to_change, int box_size):
					 
	cdef int x1,y1,i,j,k
	cdef int to_change=0
	cdef int t=0
	cdef unsigned int ind
	cdef int insert_position
	cdef int *current_param_ind
	current_param_ind=<int *>PyMem_Malloc(num_params*sizeof(int))
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			to_change=0
			for k in range(max_locations_to_change):
				if locations_to_change[0][k]==x1 and locations_to_change[1][k]==y1:
					to_change=1
					break
			if to_change==1:
				continue
			ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)
			
			for j in range(num_params):
				current_param_ind[j]=int(fitted[ind+j])
				
			if fitted[ind]>0:
				insert_position=check_present_already(param_val_ind,current_param_ind, box_size,num_params)
			
				if insert_position>=0:
					for j in range(num_params):
						param_val_ind[j][insert_position]=current_param_ind[j]
						
	PyMem_Free(current_param_ind)	
	return
	
cdef unsigned int count_unique_param_vals(int **param_val_ind, int num_params, int box_size):
	cdef int i,j
	cdef int full_box=1
	
	i=0
	for j in range(box_size):
		if param_val_ind[i][j]<0:
			return j+1
			
	return box_size
	
cdef void remove_duplicate_findings(int **discont, int discont_num, int low_indx, int low_indy, int low_indt,\
					int  high_indx, int high_indy, int high_indt):
					
	cdef int i,j
	for i in range(discont_num):
		if discont[0][i]>=low_indt and discont[0][i]<=high_indt and \
		   discont[1][i]>=low_indy and discont[1][i]<=high_indy and \
		   discont[2][i]>=low_indx and discont[2][i]<=high_indx :
		   
			for j in range(3):
				discont[j][i]=-1
	return
	
cdef double find_distance(fof_struct *temp2, fof_struct *member_struct, int num_params):
	cdef int i
	cdef double dist=0.0
	for i in range(num_params):
		dist=dist+square(temp2.fof_params[i]-member_struct.fof_params[i])
	dist=sqrt(dist)
	return dist

cdef void do_clustering_analysis(int **cluster_coords, int num_times, int num_y, int num_x,int num_params, int low_indx,\
				int low_indy, int low_indt, int high_indx, int high_indy,\
				int high_indt, double **param_vals, double [:] fitted, double max_dist, int max_locations_to_change,\
				int **locations_to_change):
	
	cdef int x1,y1,t1
	cdef fof_struct *fof
	cdef fof_struct *fof_list
	cdef int i,j
	
	cdef int cluster_numbers[2]
	cluster_numbers[0]=-1
	cluster_numbers[1]=-1
	
	t1=0
	i=0
	
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			fof=<fof_struct *>PyMem_Malloc(sizeof(fof_struct))
			make_fof_struct(x1, y1, t1, num_times, num_y, \
					num_x, num_params, fitted, param_vals,\
			 		fof)	
			if i==0:
				fof_list=fof
			else:
				cluster_numbers[0]=-1
				cluster_numbers[1]=-1
				cluster_to_add(fof_list, fof, max_dist, cluster_numbers, num_params)
				if cluster_numbers[1]<0:
					if cluster_numbers[0]>=0:
						add_member(fof_list, cluster_numbers[0],fof)
					else:
						add_clusters(fof_list, fof)
				else:
					add_member(fof_list, cluster_numbers[0],fof)
					join_clusters(fof_list, cluster_numbers[0], cluster_numbers[1])
			i=i+1
					
	cdef int num_cluster=find_cluster_num(fof_list)
	
	
	cdef int *member_number
	member_number=<int *>PyMem_Malloc(num_cluster*sizeof(int))
	find_member_number(fof_list,member_number)
	
		
	cdef int *locs_to_change
	
	locs_to_change=<int *>PyMem_Malloc(max_locations_to_change*sizeof(int))
	
	for i in range(max_locations_to_change):
		locs_to_change[i]=-1
	
	cdef int member_num, total_locs
	cdef int filled=0
	cdef int inserted
	total_locs=0
	j=0
	
	for member_num in range(4):
		for i in range(num_cluster):
			if filled==1:
				break
			if member_number[i]==member_num:
				total_locs=total_locs+member_num
				if total_locs>max_locations_to_change:
					filled=1
				else:
					locs_to_change[j]=i
					inserted=add_locs_to_change(i,locations_to_change,fof_list,max_locations_to_change)
					if inserted==0:
						filled=1
					j=j+1
	
	
	free_cluster(fof_list)
	PyMem_Free(member_number)
	PyMem_Free(locs_to_change)
	return	
	 		

cdef double remove_discontinuities(numpy.ndarray[numpy.double_t, ndim=4] cube,\
				numpy.ndarray[numpy.double_t,ndim=2] err_cube,\
				numpy.ndarray[numpy.double_t,ndim=1] model, \
				int *low_snr_loc, int ***low_freq_ind,\
				int ***upper_freq_ind,\
				double [:] fitted,double **param_val,\
				 int num_times, int num_x, int num_y, int num_params,\
				int num_freqs,int search_length, double thresh, \
				double **param_val_actual,double sys_error,\
				double spatial_smoothness_enforcer,\
				double temporal_smoothness_enforcer, int *model_param_lengths,\
				double max_dist):
	
	cdef int tot_pix,i,j
					
	tot_pix=num_times*num_y*num_x  
	
	cdef unsigned int num_val=num_times*num_y*num_x*(num_params+1)
	
	cdef double grad_chi=calc_grad_chi(fitted,param_val,num_times,num_y,num_x,num_params, \
						spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
	cdef double current_grad_chi=1e9
	cdef double *fitted_pointer
	fitted_pointer=<double *>PyMem_Malloc(num_val*sizeof(double))
	cdef double [:] fitted_copy=<double [:num_val]>fitted_pointer
	
	for i in range(num_val):
		fitted_copy[i]=fitted[i]
		
	print (grad_chi)
	
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
	
	cdef double **previous_param_inds
	
	previous_param_inds=<double **>PyMem_Malloc(discont_num*sizeof(double*))
	
	cdef int *cluster_coords[3]
	
	for i in range(3):
		cluster_coords[i]=<int *>PyMem_Malloc(box_size*sizeof(int))
		for j in range(box_size):
			cluster_coords[i][j]=-1
	
		
	for i in range(discont_num):
		previous_param_inds[i]=<double *>PyMem_Malloc(box_size*(num_params+1)*sizeof(double))
		
	cdef int max_locations_to_change=10
	cdef int *locations_to_change[2]
	
	for i in range(2):
		locations_to_change[i]=<int *>PyMem_Malloc(max_locations_to_change*sizeof(int))
		for j in range(max_locations_to_change):
			locations_to_change[i][j]=-1
	
	cdef unsigned int unique_params	
	
	for i in range(21,discont_num):
		if discont[0][i]<0 and discont[1][i]<0 and discont[2][i]<0:
			continue
			
		make_data_ready_for_discont_removal(cube,err_cube,low_snr_loc,\
					low_freq_ind,upper_freq_ind,\
					num_times,num_y,num_x,num_freqs, \
					num_params,spectrum, rms, discont[0][i],\
					discont[1][i],discont[2][i], &low_indx,\
					&low_indy, &low_indt, &high_indx, &high_indy,\
					&high_indt,search_length, param_val_ind, fitted,\
					previous_param_inds[i])
		
		
		do_clustering_analysis(cluster_coords, num_times, num_y, num_x,num_params, low_indx,low_indy,low_indt,\
					high_indx, high_indy, high_indt, param_val_actual, fitted, max_dist, max_locations_to_change,\
					locations_to_change) 
					
		get_high_confidence_param_vals(low_indx,low_indy, low_indt,high_indx,high_indy,high_indt, num_times, \
						num_y, num_x, num_params, fitted, param_val_ind,locations_to_change,\
						 max_locations_to_change, box_size)
						 
		unique_params=count_unique_param_vals(param_val_ind, num_params, box_size)
		print (unique_params)
		
		#do_bee_swarm_optimisation(spectrum, rms, model,num_freqs, param_val_ind, param_val_lengths,\
		#			 num_params, low_indx,low_indy,low_indt,high_indx,high_indy,high_indt,\
		#			 fitted, param_val_actual, sys_error,spatial_smoothness_enforcer,\
		#				temporal_smoothness_enforcer, num_x, num_y,model_param_lengths)
		remove_duplicate_findings(discont, discont_num,low_indx, low_indy, low_indt, high_indx, high_indy,high_indt)				
		break
		
	
	
		
	for i in range(3):
		PyMem_Free(discont[i])
		PyMem_Free(cluster_coords[i])
	
	
	for i in range(box_size):
		PyMem_Free(spectrum[i])
	
	
	for i in range(num_params):
		PyMem_Free(param_val_ind[i])
	
	
	for i in range(discont_num):
		PyMem_Free(previous_param_inds[i])
		
	for i in range(2):
		PyMem_Free(locations_to_change[i])
		
		
	
	current_grad_chi=calc_grad_chi(fitted,param_val,num_times,num_y,num_x,num_params, \
						spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer)
						
	
	
	if grad_chi<=current_grad_chi:
			fitted[i]=fitted_copy[i]
	else:
		grad_chi=current_grad_chi

		
					
			
	PyMem_Free(spectrum)
	PyMem_Free(rms)
	PyMem_Free(param_val_ind)
	PyMem_Free(param_val_lengths)
	PyMem_Free(previous_param_inds)
	PyMem_Free(fitted_pointer)
	return grad_chi

cdef int detect_discont_prev(double[:] fitted, double **param_val,int t0, int y, int x, int param,int num_x, \
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
	
cdef int find_median(int *x, int size):
	cdef int *sorted_array
	sorted_array=PyMem_Malloc(size*sizeof(int))
	
	memcpy(sorted_array, x, size*sizeof(int)) 
	
	cdef int i
	cdef int temp,indx
	cdef int min_indx
	cdef int min1
	
	for i in range(size):
		min_indx=i
		for indx in range(i,size):
			temp=sorted_array[indx]
			if temp<sorted_array[min_indx]:
				min_indx=indx
		temp=sorted_array[min_indx]
		sorted_array[min_indx]=sorted_array[i]
		sorted_array[i]=temp
	
	PyMem_Free(sorted_array)
	return sorted_array[size//2]
	
	
cdef int find_mad(int *x,int median, int size):
	return 0
#TODO The function below may have issues. Check. 

cdef int detect_discont(double[:] fitted, double **param_val,int t0, int y, int x, int param,int num_x, \
			int num_y, int num_times, int num_params, int search_length,\
			 int axis, double thresh,int reverse):
	
	cdef int t,x1,y1,ind, ind1, ind5,j
	cdef double sum1, sum2, mean,std, all_mean, ratio
	cdef int low_indx, low_indy, high_indx,high_indy
	
	sum1=0.0
	sum2=0.0
	
	ind5=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x*(num_params+1)+num_params
	if fitted[ind5]<-0.2:
		return 0
			
	cdef double param_value
	
	param_value=param_val[param][int(fitted[ind5-num_params+param])]
	
	
	cdef int num
		
	
	low_indx=x-search_length//2
	high_indx=x+search_length//2
		
	low_indy=y-search_length//2
	high_indy=y+search_length//2
	
	while low_indx<0:
		low_indx=low_indx+1
	
	while low_indy<0:
		low_indy=low_indy+1
		
	while high_indx>num_x-1:
		high_indx=high_indx-1
	
	while high_indy>num_y-1:
		high_indy=high_indy-1
		
	if (high_indy-low_indy<search_length) or (high_indx-low_indx<search_length):
		return 0
	
	
	cdef int *for_median
	
	for_median=<int *>PyMem_Malloc(search_length*search_length*sizeof(int))	
	
	j=0
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			ind5=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind5]>0.0:
				ind=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x1*(num_params+1)+param
				for_median[j]=int(fitted[ind])
				j=j+1
	cdef int median=find_median(for_median,j)
	cdef int mad=find_mad(for_median,median,j)
	
	j=0
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			ind5=t0*num_y*num_x*(num_params+1)+y*num_x*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind5]>0.0:
				if mad<1e-4 and absolute(for_median[j]-median)<1e-4:
					return 0
				elif mad<1e-4 and absolute(for_median[j]-median)>1e-4:
					return 1
				elif for_median[j]>thresh*mad+median:
					return 1
	
				j=j+1
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

cdef int detect_low_snr_freqs(double *spectrum,double *rms, double rms_thresh, int *pos, int low_ind, int high_ind,int num_freqs):
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
					grad_x[param1]=(absolute(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])+\
							absolute(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])]))/2
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
					grad_y[param1]=(absolute(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])+\
							absolute(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])]))/2
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
					grad_t[param1]=(absolute(param_val[param1][int(fitted[ind])]-\
							param_val[param1][int(fitted[ind1])])+\
							absolute(param_val[param1][int(fitted[ind2])]-\
							param_val[param1][int(fitted[ind])]))/2
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
				double spatial_smoothness_enforcer=0.0, double temporal_smoothness_enforcer=0.0):
	cdef double tot_chi=0.0
	cdef int t,y1,x1,ind,param1
	cdef int ind1,ind2,ind3
	
	cdef double *grad_x
	cdef double *grad_y
	cdef double *grad_t
	grad_x=<double *>PyMem_Malloc(num_params*sizeof(double))
	grad_y=<double *>PyMem_Malloc(num_params*sizeof(double))
	grad_t=<double *>PyMem_Malloc(num_params*sizeof(double))
	
	
	for l in range(num_params):
		grad_x[l]=0.0
		grad_y[l]=0.0
		grad_t[l]=0.0
	
	
	
	for t in range(num_times):
		for y1 in range(num_y):
			for x1 in range(num_x):
				ind=t*num_y*num_x*(num_params+1)+y1*num_x*(num_params+1)+x1*(num_params+1)+num_params
				
				
				if fitted[ind]>-0.2:
					tot_chi=tot_chi+fitted[ind]
					calc_grad(fitted,param_val,num_times,num_y,num_x,t,y1,x1,num_params,grad_x,grad_y,grad_t)
					
					if num_x>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+spatial_smoothness_enforcer*square(grad_x[param1])
					if num_y>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+spatial_smoothness_enforcer*square(grad_y[param1])
					
					if num_times>1:
						for param1 in range(num_params):
							tot_chi=tot_chi+temporal_smoothness_enforcer*square(grad_t[param1])
				
	PyMem_Free(grad_x)
	PyMem_Free(grad_y)
	PyMem_Free(grad_t)						
	return tot_chi
	
cdef void make_cube_fit_ready(int num_times,int num_y, int num_x,int num_freqs, double [:,:]err_cube,\
			double [:,:,:,:]cube, double *freqs1, double lower_freq,\
			double upper_freq, int ***low_freq_ind, int ***upper_freq_ind, int min_freq_num,\
			int num_params, double [:] fitted, int *pos, double rms_thresh):

	cdef int t, i, y1, x1,j, low_ind, high_ind, high_snr_freq_num,l, ind5, ind3
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
	
cdef double calc_red_chi_all_pix(int num_times, int num_freqs, int num_y, int num_x, int num_params,int ***low_freq_ind,\
						int ***upper_freq_ind, numpy.ndarray[numpy.double_t,ndim=4] cube,\
						numpy.ndarray[numpy.double_t,ndim=2] err_cube,numpy.ndarray[numpy.double_t,ndim=1] model, double sys_error,\
						int *pos,double [:] fitted, double *freqs1, double lower_freq,\
						double upper_freq, int first_pass, double rms_thresh, int min_freq_num, int *param_lengths1,\
						int *param_ind, double **param_val,double spatial_smoothness_enforcer,\
						double temporal_smoothness_enforcer, int search_length, double discont_thresh, double max_dist):
						
						
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
	
	
					
	
						
	
				
	cdef double grad_chi=remove_discontinuities(cube, err_cube, model, pos, low_freq_ind, upper_freq_ind, fitted1,param_val,num_times,num_x, num_y, \
				num_params,num_freqs,search_length,discont_thresh, param_val, sys_error,spatial_smoothness_enforcer,\
						temporal_smoothness_enforcer,param_lengths1, max_dist)		
						
	cdef unsigned int num_val=num_times*num_x*num_y*(num_params+1)
	
	for i in range(num_val):
		fitted[i]=fitted1[i]		
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
				double spatial_smoothness_enforcer, double temporal_smoothness_enforcer,\
				double frac_tol, int max_iter, int search_length, double discont_thresh, double max_dist_parameter_space):
				
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
						search_length,discont_thresh, max_dist_parameter_space)
						
						
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
				
