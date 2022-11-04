cimport numpy
import numpy as np
from libc.math cimport sqrt,exp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 
import cython
#@cython.boundscheck(False)
#@cython.wraparound(False)

cdef struct fof_struct:
	int x
	#int y
	#int t
	#double *fof_params
	fof_struct *cluster_forward
	fof_struct *cluster_backward
	fof_struct *member_forward
	fof_struct *member_backward
	
cdef void make_fof_struct(int x0, fof_struct *fof):
	fof.cluster_forward=NULL
	fof.cluster_backward=NULL
	fof.member_forward=NULL
	fof.member_backward=NULL
	#cdef double *fof_params
	#fof_params=<double *>PyMem_Malloc(num_params*sizeof(double))
	
	#for i in range(num_params):
	#	fof_params[i]=param_vals[i][int(fitted[ind+i])]
	
	#fof->fof_params=fof_params
	
	fof.x=x0
	#fof->y=y0
	#fof->t=t0
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
	
cdef int absolute(int x):
	if x<0:
		return -x
	return x
	
cdef find_distance(fof_struct *temp2, fof_struct *member_struct):
	return absolute(temp2.x-member_struct.x)
	
cdef void cluster_to_add(fof_struct *fof_list, fof_struct *member_struct, double max_dist, int *cluster_numbers):
	
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
			dist=find_distance(temp2,member_struct)
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

cdef void print_cluster(fof_struct *fof_list):
	
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
			print (temp2.x)
			temp2=temp2.member_forward
		temp1=temp1.cluster_forward
		i=i+1
	return
	

cpdef void gen_list(numpy.ndarray[numpy.int_t,ndim=1]a, int num, int max_dist):
	cdef int i
	
	cdef fof_struct *fof
	cdef fof_struct *fof_list
	
	cdef int cluster_numbers[2]
	cluster_numbers[0]=-1
	cluster_numbers[1]=-1
	
	
	for i in range(num):
		fof=<fof_struct *>PyMem_Malloc(sizeof(fof_struct))
		make_fof_struct(a[i],fof)
		if i==0:
			fof_list=fof
			#add_first_cluster(fof_list, fof)
		else:
			cluster_numbers[0]=-1
			cluster_numbers[1]=-1
			cluster_to_add(fof_list, fof, max_dist, cluster_numbers)
			if cluster_numbers[1]<0:
				if cluster_numbers[0]>=0:
					add_member(fof_list, cluster_numbers[0],fof)
				else:
					add_clusters(fof_list, fof)
			else:
				add_member(fof_list, cluster_numbers[0],fof)
				join_clusters(fof_list, cluster_numbers[0], cluster_numbers[1])
			
	print_cluster(fof_list)
	free_cluster(fof_list)
	return	
