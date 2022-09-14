import numpy as np
from spectrum import Spectrum
import cython_functions_for_fast_computation as cfunc
import h5py
from scipy.ndimage import gaussian_filter
from itertools import product



def check_at_boundary(x0,\
		      y0,\
		      numx,\
		      numy,\
		      num_params,\
		      fitted):
	'''
	Here boundary DOES NOT mean the boundary of the 
	image cube. It just means the edge at after which 
	the data could not be fitted based on the current
	user specified parameters.
	
	x0: x position
	y0: y position
	numx: total x length
	numy: total y length
	num_params: number of unkwnon parameters
	fitted: the array containing the current fit results
	
	This function does not check the validity of x and y 
	inputs, as the function calling it will never send
	an invalid coordinate. And hence putting that check 
	will result in unnecessary overhead.
	'''
	
	
	x=[x0-1,x0+1,x0,x0,x0-1,x0+1,x0-1,x0-1]
	y=[y0,y0,y0-1,y0+1,y0+1,y0+1,y0-1,y0+1]
	
	for x1,y1 in zip(x,y):
		ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
		if fitted[ind]<0:
			return 1
	
	return 0


def detect_discont(fitted,\
		   param_val,\
		   x,\
		   y,\
		   numx,\
		   numy,\
		   num_params,\
		   search_length,\
		   thresh):
	'''
	This function detects pixel level discontinuities
	based on median-rms approach. I have chosen to not take
	the mad because since this is a index based fitting
	method, sometimes, the mad can give zero, resulting in
	false discontinutites and also 0/0 values.
	
	fitted: the array containing the current fit results
	param_val: the actual parameter values in physical units.
		    Not used now. 
	x,y: Spatial coordinates
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	search_length: Averaging interval for detecting discontinuity
	thresh:  Threshold in terms of std, above which discontinuity
		 is identified.
		 
	I take an area around x+-search_length//2, subject to the boundary 
	constraints of the cube. If the length is less than search length,
	I return. For every parameter, I calculate median and std using the
	median. If the (value-median)/std> thresh, I detect a discontinuity.
	And I return, as there is no further need to search other parameters.
	'''
	
	low_indx=max(0,x-search_length//2)
	low_indy=max(0,y-search_length//2)

	high_indx=min(numx-1,x+search_length//2)
	high_indy=min(numy-1,y+search_length//2)
	
	if high_indx-low_indx<search_length or high_indy-low_indy<search_length:
		return 0

	for_median=np.zeros(int((search_length+1)**2))
	
	
	for param in range(num_params):
		j=0
		for_median[:]=np.nan
		for y1 in range(low_indy,high_indy+1):
			for x1 in range(low_indx,high_indx+1):
				ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
				if fitted[ind]<0:
					continue
				ind=y1*numx*(num_params+1)+x1*(num_params+1)+param
				for_median[j]=param_val[param][int(fitted[ind])]
				j+=1
		if j<search_length:
			return 0
		median=np.nanmedian(for_median)
		mad=np.nanmean(np.absolute(for_median-median))  ### note nanmean
		
		ind_center=y*numx*(num_params+1)+x*(num_params+1)+param
		
		at_boundary=check_at_boundary(x,y,numx,numy,num_params,fitted)
		
		if mad<1e-4 and median<1e-4:
			return 0
		elif mad<1e-4 and abs(median-fitted[ind_center])>1e-4 and at_boundary!=1:
			return 1
		elif abs(median-param_val[param][int(fitted[ind_center])])>thresh*mad and at_boundary!=1:
			return 1
		
		
def list_discont(fitted,\
		 param_val,\
		 numx,\
		 numy,\
		 num_params,\
		 search_length,\
		 thresh):
	'''
	This function calls detect discont for every x,y and finally
	returns all the detected discontinuitites.
	x,y: Spatial coordinates
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	search_length: Averaging interval for detecting discontinuity
	thresh:  Threshold in terms of std, above which discontinuity
		 is identified.
	'''
	discont=[]
	for y in range(numy):
		for x in range(numx):
			ind5=y*numx*(num_params+1)+x*(num_params+1)+num_params
			if fitted[ind5]<0:
				continue
			discont_found=detect_discont(fitted, param_val,x,y,numx,numy,num_params,search_length,thresh)
			if discont_found==1:
				discont.append([x,y])
	return discont




def get_clusters(fitted,\
		 low_indx,\
		 low_indy, \
		 high_indx,\
		 high_indy,\
		 numx,\
		 numy,\
		 num_params,\
		 max_dist_parameter_space,\
		 param_lengths):
	'''
	Here I implement a cluster finder using FOF algorithm.
	The code finds cluster in the parameter space. Now I 
	had two choices. I can either use the actual parameter
	values or use the fit indices. If I choose the parameter
	values then I have to normalise them somehow so that I
	can use a single spatial_smoothness_enforcer. Having a
	different multiplier for each parameter is extremely
	crude in my opinion. if I normalise the values using the 
	mean, then I would minimise spread on the parameter whose
	range is the least. Because the ratio/difference with respect
	to the other parameters would be less for that. Hence, if
	there is a discontinuity in that parameter only, it will
	not be detected. I thought of using the length of the
	model paramters. Since they are indices, then they are on
	same sclae and hence there is no need to normalise. However
	here also I faced an issue. The parameter which is the longest
	gets the largest weight during this, which is again I did not want. 
	So while getting the distance along an parameter axes, I first
	take the difference between the indices of the parameter
	and then strtech that by the ratio of the maximum parameter length
	and the length of the parameter where the search is being done.
	I find that this creates a smooth map in all parameters.
	No ill effect of this kind of normalisation and distance finding
	has been found yet.  
	
	fitted: the array containing the current fit results
	low_indx,low_indy: the corodinate of the bottom-left corner of
	                   the box where the search should be done
	high_indx,high_indy: the corodinate of the top-right corner of
	                     the box where the search should be done
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	max_dist_parameter_space: The maximum distance between two cluster
	                          in the parameter space. The distance is
	                          calculated according to the discussion above.
	param_lengths: The length of the parameters the user has provided.
			This is read from the model cube provided by the user.
	'''
	
	cluster_list=[]
	max_param_length=np.max(param_lengths)
	
	j=0
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			ind0=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind0]<0:
				continue
			j+=1
			cluster_member=False
			clusters=[]
			for n,cluster in enumerate(cluster_list):
				for member in cluster:
					x0=member[0]
					y0=member[1]
					dist=0
					for param in range(num_params):
						ind1=y0*numx*(num_params+1)+x0*(num_params+1)+param
						ind0=y1*numx*(num_params+1)+x1*(num_params+1)+param
						dist=dist+((fitted[ind1]-fitted[ind0])*max_param_length/param_lengths[param])**2
					dist=np.sqrt(dist)
					if dist<max_dist_parameter_space:
						cluster_member=True
						clusters.append(n)
						break
			if len(clusters)==0:
				cluster_list.append([[x1,y1]])
			elif len(clusters)==1:
				cluster_list[clusters[0]].append([x1,y1])
			else:
				cluster_list[clusters[0]].append([x1,y1])
				for m in cluster_list[clusters[1]]:
					cluster_list[clusters[0]].append(m)
				del cluster_list[clusters[1]]
	
	return cluster_list
	
def get_points_to_remove(cluster_list):
	'''
	Here I am passed a cluster list, which was created in the box
	where at least discontituity has been detected. Because of
	memory constraints, I choose not to try to replace the values of
	all the detected points. I choose to replace a maximum of 10
	points. So I start with the clusters which ahs lowest number of member,
	which is generally 1. Then I start putting in points from that cluster.
	I always include all points from a cluster for removal or donot include
	any at all. 
	
	cluster_list: List of all clusters detected and their members
	'''
	member_num=[]
	for cluster in cluster_list:
		member_num.append(len(cluster))
	
	min_member=int(min(member_num))
	max_member=int(min(max(member_num)//1.2,4))
	## In earlier versions, max_member was always set to 4. 
	points_to_remove=[]
	max_points_to_remove=10
	j=0
	
	for current_member_number in range(min_member,max_member+1):
		for n,cluster in enumerate(cluster_list):
			if member_num[n]==current_member_number:
				if j+member_num[n]>max_points_to_remove:
					continue
				for member in cluster:
					points_to_remove.append(member)
					j+=1
	return points_to_remove
	
	
def find_new_params(points_to_remove,\
		     fitted,\
		     model,\
		     spectrum,\
		     sys_error,\
		     rms,\
		     param_val,\
		     numx,\
		     numy,\
		     num_params,\
		     low_indx,\
		     low_indy,\
		     high_indx,\
		     high_indy,\
		     param_lengths,\
		     num_freqs,\
		     low_freq_ind,\
		     upper_freq_ind,\
		     rms_thresh,\
		     smoothness_enforcer):
	'''
	This function finds new parameter values for the points in
	the list returned by points_to_remove function. I take the
	parameter values of the neighbours of each point, unless 
	neighbour was itself in the points_to_remove list. Then I
	iterate through each point in points_to_Remove list and 
	calculate the grad_chi_square with the values of its neighbours.
	The grad_chi_square has been implemented in Cython. If I
	see that the grad_chi_square has decreased, I keep the new
	parameter values.
	
	points_to_remove: list of the points to be replaced returned 
	                  by get_points_to_remove function
	fitted: the array containing the current fit results
	model: model cube supplied by the user
	spectrum: Observed image cube provided by user. Shape: numy,numx,num_freqs
	sys_error: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	rms: Rms of the image cube. Shape= num_freqs
	param_val: Actual values of the parameters
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	low_indx,low_indy: the corodinate of the bottom-left corner of
	                   the box where the search should be done
	high_indx,high_indy: the corodinate of the top-right corner of
	                     the box where the search should be done
	param_lengths: Lengths of the parameters in the supplied model cube
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	'''
	
	new_params=[]
	points_changed=0
	for m,point in enumerate(points_to_remove):
		changed=False
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		new_params.append([])
		for i in range(ind,ind+num_params+1):
			new_params[m].append(fitted[i])
		inds=[]
		for i in range(-2,3):
			for j in range(-2,3):
				x1=x0+j
				y1=y0+i
				if [x1,y1] not in points_to_remove:
					ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
					if fitted[ind]>0:
						inds.append(fitted[ind-num_params:ind].tolist())
		low_freq=low_freq_ind[y0*numx+x0]
		upper_freq=upper_freq_ind[y0*numx+x0]
		spectrum1=np.ravel(spectrum[y0-low_indy,x0-low_indx,:])
		grad_chi_square=cfunc.calc_grad_chisquare(max(low_indx, x0-2),max(low_indy,y0-2),min(high_indx,x0+2),min(high_indy,y0+2), \
					numx,numy,num_params, fitted, param_lengths, smoothness_enforcer)	
		for n,param_ind in enumerate(inds):
			model_ind=0
			for k in range(num_params):
				product=1
				for l in range(k+1,num_params):
					product=product*param_lengths[l]
				model_ind+=param_ind[k]*product*num_freqs
				fitted[y0*numx*(num_params+1)+x0*(num_params+1)+k]=param_ind[k]
			
			chi_square=cfunc.calc_chi_square(spectrum1,rms,sys_error,model[int(model_ind):],low_freq,upper_freq,rms_thresh)
			fitted[y0*numx*(num_params+1)+x0*(num_params+1)+num_params]=chi_square
			grad_chi_square_temp=cfunc.calc_grad_chisquare(max(low_indx, x0-2),max(low_indy,y0-2),min(high_indx,x0+2),min(high_indy,y0+2),\
						 numx,numy,num_params, fitted,param_lengths,smoothness_enforcer)		
			if grad_chi_square_temp<grad_chi_square:
				changed=True
				grad_chi_square=grad_chi_square_temp
				for param1 in range(num_params):
					new_params[m][param1]=param_ind[param1]
				new_params[m][num_params]=chi_square
		if changed==True:
			for k in range(num_params+1):
				fitted[y0*numx*(num_params+1)+x0*(num_params+1)+k]=new_params[m][k]
			points_changed+=1
	return new_params,points_changed
					
def remove_discont(spectral_cube, \
			err_cube,\
			fitted,\
			param_val,\
			numx,\
			numy,\
			num_params,\
			smooth_lengths,\
			thresh,\
			max_dist_parameter_space,\
			 model,param_lengths,\
			 sys_err,\
			 num_freqs,\
			 low_freq_ind,\
			 upper_freq_ind,\
			 rms_thresh,\
			 smoothness_enforcer,\
			 max_iter=5):

	'''
	This function is a wrapper for the all the functions needed to remove
	the pixel scale discontinuities. The user calls this function and this
	calls others as needed.
	
	spectral_cube: Observed spectral cube, Shape: num_times, num_y, num_x, num_freqs
	err_cube: Image rms cube, shape: num_times, num_freqs
	fitted: the array containing the current fit results
	param_val: Actual values of the parameters
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_lengths: The list of smoothing lengths to be used for finding
			 discontituiny
	thresh: Threshold used to detect discontinuity.
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	param_lengths: Lengths of the parameters in the supplied model cube
	sys_error: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	max_iter: Optional parameter. Maximum number of iterations for which the search
		  and removal process is repeated. If no points are changed in any
		  iteration search and removal process for that smoothing length
		  is stopped.	
	'''
	
	rms=np.ravel(err_cube[0,:])
	for search_length in smooth_lengths:
		iter1=0
		points_changed=0
		while iter1<max_iter:
			points_changed=0
			discont_list=list_discont(fitted, param_val,numx,numy,num_params,search_length,thresh)
			
			for point_num,discont in enumerate(discont_list):
				x=discont[0]
				y=discont[1]
				
				if x<0 and y<0:
		
					continue
				low_indx=max(0,x-search_length//2)
				low_indy=max(0,y-search_length//2)

				high_indx=min(numx-1,x+search_length//2)
				high_indy=min(numy-1,y+search_length//2)
				
				if high_indx-low_indx<search_length or high_indy-low_indy<search_length:
					continue
				
				spectrum=spectral_cube[0,low_indy:high_indy+1,low_indx:high_indx+1,:]
				
				
				clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space,param_lengths)
				
				
				points_to_remove=get_points_to_remove(clusters)
				
				points_changed+=remove_all_points(points_to_remove,spectrum,rms,sys_err, fitted, param_val,numx,numy,num_params,\
							search_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy,\
							param_lengths,num_freqs,low_freq_ind,upper_freq_ind, rms_thresh,smoothness_enforcer)
				
				for discont1 in discont_list[point_num+1:]:
					if discont1[0]>=low_indx and discont1[0]<=high_indx	and discont1[1]>=low_indy and discont1[1]<=high_indy:
						discont1[0]=-1
						discont1[1]=-1	
					
			iter1+=1
			if points_changed==0:
				break
		
	return		

def remove_all_points(points_to_remove,\
			spectrum,\
			rms,\
			sys_err,\
			fitted,\
			param_val,\
			numx,\
			numy,\
			num_params,\
			search_length,\
			thresh,\
			max_dist_parameter_space,\
			model,\
			low_indx,\
			low_indy,\
			high_indx,\
			high_indy,\
			param_lengths,\
			num_freqs,\
			low_freq_ind,\
			upper_freq_ind,\
			rms_thresh,\
			smoothness_enforcer):
	'''
	This function first calls get_new_params and then decides if the
	found points can be kept considering the full box.
	
	points_to_remove: points_to_remove obtained from get_points_to_remove
			   function 
	spectrum: Observed spectral cube, Shape: num_times, num_y, num_x, num_freqs
	rms: Image rms cube, shape: num_freqs
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	fitted: the array containing the current fit results
	param_val: Actual values of the parameters
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_length: smoothing length to be used for finding
			 discontituiny
	thresh: Threshold used to detect discontinuity.
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	low_indx,low_indy: the corodinate of the bottom-left corner of
	                   the box where the search should be done
	high_indx,high_indy: the corodinate of the top-right corner of
	                     the box where the search should be done
	param_lengths: Lengths of the parameters in the supplied model cube
	
	
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	'''

	old_params=[]
	for m,point in enumerate(points_to_remove):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		old_params.append([])
		for i in range(ind,ind+num_params+1):
			old_params[m].append(fitted[i])

	grad_chi_square_old=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, fitted,param_lengths,smoothness_enforcer)	
	
	new_params,points_changed=find_new_params(points_to_remove,fitted,model,spectrum,sys_err,rms,param_val,numx,numy,num_params,\
							 low_indx,low_indy,high_indx,high_indy,param_lengths,num_freqs, low_freq_ind,\
							 upper_freq_ind, rms_thresh,smoothness_enforcer)
	
	for m,point in enumerate(points_to_remove):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		for k,val in enumerate(new_params[m]):
			fitted[ind+k]=val
	grad_chi_square_new=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, fitted,param_lengths,smoothness_enforcer)	
	if grad_chi_square_new>grad_chi_square_old:
		for m,point in enumerate(points_to_remove):
			x0=point[0]
			y0=point[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for k,val in enumerate(old_params[m]):
				fitted[ind+k]=val
		return 0
			
	return	points_changed
	
def get_new_param_inds(spectrum,\
			rms,\
			model,\
			min_params,\
			max_params,\
			param_lengths,\
			new_params_temp,\
			low_freq_ind,\
			upper_freq_ind,\
			rms_thresh,\
			num_params, \
			num_freqs,\
			sys_err):
	'''
	This is the function which is used to find new parameter values
	when search is done in the cluster space, irerspective of whether
	a discontinuity is present or not. The important thing to note is
	that the cython code actually has a function with name same as this.
	If used, that can make this part of the code ~11 times faster. 
	The only reason I am not using that is that there I had to hardcode
	the number of fitting parameters in the for loops. The number of for
	loops is exactly equal to the number of fitted parameters. While I
	would try to remove this and write something more general, if you want
	blazing fast speed and can handle code changing, please consider using
	that function. On the otherhand, this function is a general function
	and can handle any number of unknowns. 
	
	spectrum: 1D spectrum
	rms: image rms 
	model: user supplied model cube
	min_params: The minimum parameter indices possible. Shape: num_params
	max_params: The maximum parameter indices possible. Shape: num_params
	param_lengths: Lengths of the parameters in the supplied model cube  
	new_params_temp: Holder of new found parameter values
	low_freq_ind: Lowest frequency to be used for chi_square calculation
	upper_freq_ind: Highest frequency to be used for chi_square_calculation
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	num_params: Number of fitted parameters
	num_freqs: Number of frequencies in the image cube.
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	'''
	possible_param_indices=[None]*num_params
	for i in range(num_params):
		possible_param_indices[i]=np.arange(min_params[i],max_params[i]+1)
	
	
	chisq=np.array([1e9])
	for i in product(*possible_param_indices):
		param_ind=np.array(i,dtype=np.intc)
		cfunc.get_new_param_inds_general(spectrum,rms,model,param_ind,param_lengths, new_params_temp,low_freq_ind,\
							upper_freq_ind,rms_thresh,num_params, num_freqs,sys_err,chisq)
	return
	
def remove_big_clusters(clusters,\
			 cluster1,\
			 cluster2,\
			 spectral_cube,\
			 err_cube,\
			 sys_err,\
			 fitted, \
			 param_val,\
			 numx,\
			 numy,\
			 num_params,\
			 smooth_length,\
			 thresh,\
			 max_dist_parameter_space,\
			 model,\
			 low_indx,\
			 low_indy,\
			 high_indx,\
			 high_indy,\
			 min_params1,\
			 max_params1,\
			 num_freqs,\
			 low_freq_ind,\
			 upper_freq_ind,\
			 rms_thresh,\
			 param_lengths,\
			 smoothness_enforcer):
								
	'''
	This function tries to find new parameters for a small cluster
	close to the parameter values of the biggest cluster in that
	box.
	
	clusters: List of all clusters detected in the box
	cluster1: Members of the biggest cluster of the box
	cluster2: Members of the smaller cluster which would be analysed
		  in this call
	spectral_cube: Observed spectrum. Shape-num_timesx num_y x num_x x num_freqs
			At this moment num_times>1 has not been implemented
	err_cube: Image rms cube. Shape= num_times x num_freqs. At this moment 
		   num_times>1 has not been implemented.
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	fitted: array containing current fitted parameter values
	param_val: Array containing the actual parameter values
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_length: smoothing length to be used
	thresh: Threshold used to detect discontinuity (not used probably)
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	low_indx,low_indy: the corodinate of the bottom-left corner of
	                   the box where the search should be done
	high_indx,high_indy: the corodinate of the top-right corner of
	                     the box where the search should be done
	min_params1, max_params1: List of the minimum and maximum allowed
				   allowed values of all parameters      
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency               
	param_lengths: Lengths of the parameters in the supplied model cube
	'''

	old_params=[]
	for m,point in enumerate(cluster2):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		old_params.append([])
		for i in range(ind,ind+num_params+1):
			old_params[m].append(fitted[i])	
			
	grad_chi_square_old=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy,numx,numy,num_params,fitted,param_lengths,smoothness_enforcer)
	
	rms=np.ravel(err_cube[0,:])
	new_params_temp=np.ravel(np.ones(num_params+1)*(-1))
	for m,point in enumerate(cluster2):
		x0=point[0]
		y0=point[1]
		spectrum=np.ravel(spectral_cube[0,y0,x0,:])
		new_params_temp[:]=-1
		freq_ind=y0*numx+x0
		get_new_param_inds(spectrum,rms,model,min_params1,max_params1,param_lengths, new_params_temp,low_freq_ind[freq_ind],\
							upper_freq_ind[freq_ind],rms_thresh,num_params, num_freqs,sys_err)	
		if new_params_temp[0]>0:
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for i in range(ind,ind+num_params+1):
				fitted[i]=new_params_temp[i-ind]
				
		
	grad_chi_square_new=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, fitted,param_lengths,smoothness_enforcer)
	
	
	
	if grad_chi_square_old<grad_chi_square_new:
		for m,point in enumerate(cluster2):
			x0=point[0]
			y0=point[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for k,val in enumerate(old_params[m]):
				fitted[ind+k]=val
		return False
	
	return True
		
	
def get_total_members(clusters):
	'''
	Get total number of members in a cluster
	
	clusters: list of clusters
	'''
	member_num=0
	for cluster in clusters:
		member_num+=len(cluster)
	return member_num
	
def form_subcubes_with_gradients(numx,\
				  numy,\
				  num_params,\
				  fitted,\
				  smooth_length,\
				  param_lengths,\
				  smoothness_enforcer):
	'''
	This function first creates all the subcubes in the image
	of size smooth_length and then calculats the gradient of 
	each one of them.
	
	numx,numy: Length along X and Y coordinates of image cube
	num_params: Number of fitted parameters
	fitted: array containing current fitted parameter values
	smooth_length: smoothing length to be used
	param_lengths: Lengths of the parameters in the supplied model cube
	'''
	cube_vals=np.ones((numy*numx,3))*(-1)
	j=0
	for y in range(0,numy):
		for x in range(0,numx):
			low_indx=max(0,x-smooth_length//2)
			low_indy=max(0,y-smooth_length//2)

			high_indx=min(numx-1,x+smooth_length//2)
			high_indy=min(numy-1,y+smooth_length//2)
			
			
			if high_indx-low_indx+1<smooth_length or high_indy-low_indy+1<smooth_length:
				j+=1
				continue
				
			
			cube_vals[j,0]=x
			cube_vals[j,1]=y
			grad=cfunc.calc_gradient_wrapper(x,y,fitted, numx,numy,num_params,param_lengths,smoothness_enforcer)
			cube_vals[j,2]=grad
			j+=1
	return cube_vals
	
	
	
def remove_overlapping_subcubes(cube_vals,x0,y0,smooth_length,numy,numx):
	'''
	Here I remove the subcubes for which one searh has already been done.
	This is because, even we are concerned with the point (x0,y0), we 
	search for a box of size smooth_length around it. So there is no point
	is suppose search in a box centred around (x0-1,y0-1).
	
	cube_vals: The subcube list which has all the subcubes and their gradients
		    calculated
	x0,y0: coordinate around which the search has happened
	smooth_length: smooth length determines the box size for which search has 
			happened and should be used for finding overlaps.
	numx,numy: Length along X and Y coordinates of image cube
	'''
	j=0
	for y in range(0,numy):
		for x in range(0,numx):
			low_indx=max(0,x-smooth_length//2)
			low_indy=max(0,y-smooth_length//2)

			high_indx=min(numx-1,x+smooth_length//2)
			high_indy=min(numy-1,y+smooth_length//2)
			
			
			if high_indx-low_indx+1<smooth_length or high_indy-low_indy+1<smooth_length:
				j+=1
				continue
			if x0>=low_indx and x0<=high_indx and y0>=low_indy and y0<=high_indy:
				cube_vals[j,2]=-1.0	
			j+=1	
	return		

def smooth_param_maps(spectral_cube,\
		       err_cube, \
		       fitted,\
		       param_val,\
		       numx,\
		       numy,\
		       num_params,\
		       smooth_lengths,\
		       thresh,\
		       max_dist_parameter_space,\
		       model,\
		       param_lengths,\
		       sys_err,\
		       num_freqs,\
		       low_freq_ind,\
		       upper_freq_ind,\
		       rms_thresh,\
		       smoothness_enforcer,\
		       max_iter=5):
	
	'''
	This function is a wrapper for the all the functions needed to do
	the cluster wise analysis. The user calls this function and this
	calls others as needed.
	
	spectral_cube: Observed spectral cube, Shape: num_times, num_y, num_x, num_freqs
	err_cube: Image rms cube, shape: num_times, num_freqs
	fitted: the array containing the current fit results
	param_val: Actual values of the parameters
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_lengths: The list of smoothing lengths to be used for finding
			 discontituiny
	thresh: Threshold used to detect discontinuity.
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	param_lengths: Lengths of the parameters in the supplied model cube
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	max_iter: Optional parameter. Maximum number of iterations for which the search
		  and removal process is repeated. If no points are changed in any
		  iteration search and removal process for that smoothing length
		  is stopped.	
	'''
	
	j=0
	for smooth_length in smooth_lengths:
		iter1=0
		changed_points=0
		while iter1<max_iter:
			subcubes=form_subcubes_with_gradients(numx,numy,num_params,fitted,smooth_length,param_lengths,smoothness_enforcer)
			grads=subcubes[:,2]
			sorted_indices=np.argsort(grads)[::-1]
			subcubes[:,2]=grads
			for sort_ind in sorted_indices:
				if subcubes[sort_ind,2]<0:
					continue
				x=int(subcubes[sort_ind,0])
				y=int(subcubes[sort_ind,1])
				
				low_indx=int(max(0,x-smooth_length//2))
				low_indy=int(max(0,y-smooth_length//2))

				high_indx=int(min(numx-1,x+smooth_length//2))
				high_indy=int(min(numy-1,y+smooth_length//2))
				
				
			
				if high_indx-low_indx+1<smooth_length or high_indy-low_indy+1<smooth_length:
					continue	
				
				clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space,param_lengths)
				
				if len(clusters)<=1:
					continue	
				tot_member=get_total_members(clusters)
				if tot_member<smooth_length**2:
					continue
				
				
							
				
				
				cluster_len=len(clusters)
				if cluster_len<=1:
					return	
				member_num=[]
				for cluster in clusters:
					member_num.append(len(cluster))	
				
				member_num=np.array(member_num)
				sorted_pos=np.argsort(member_num)[::-1]
				
				cluster1=clusters[sorted_pos[0]]
				len_cluster1=len(cluster1)

				min_params1=np.ravel(np.zeros(num_params,dtype=np.intc))
				max_params1=np.ravel(np.zeros(num_params,dtype=np.intc))
				for param in range(num_params):
					ind=[]
					for i in range(len_cluster1):
						x1=cluster1[i][0]
						y1=cluster1[i][1]
						ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
						ind.append(fitted[ind1])
					ind=np.array(ind)
					min_params1[param]=max(0,int(np.min(ind))-2)
					max_params1[param]=min(int(np.max(ind))+2,param_lengths[param]-1)
					del ind	
				
				if member_num[sorted_pos[0]]==member_num[sorted_pos[1]]:
					continue
				for cluster_num in sorted_pos[1:]:			
					changed=remove_big_clusters(clusters,clusters[sorted_pos[0]],clusters[cluster_num],spectral_cube,err_cube,\
												sys_err,fitted, param_val,numx,numy,num_params,\
											smooth_length,thresh,max_dist_parameter_space, model, low_indx,\
											low_indy,high_indx,high_indy,min_params1,max_params1,\
											num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,param_lengths,\
											smoothness_enforcer)
					if changed==True:
						changed_points+=1
				remove_overlapping_subcubes(subcubes,x,y,smooth_lengths[0],numy,numx)
			if changed_points==0:
				break
			iter1+=1

def create_smoothed_model_image(low_freq_ind,\
				 high_freq_ind,\
				 num_x, \
				 num_y, \
				 num_params, \
				 num_freqs, \
				 resolution,\
				 model,\
				 fitted, \
				 smoothed_model_cube, \
				 low_indx, \
				 low_indy, \
				 high_indx,\
				 high_indy):
	'''
	This function first creates a model cube using the parameter information.
	In regions where fitting has not been done due to user choices, it uses
	the actual image values. Then I convolve each frequency image using the 
	appropriate gaussain. For convolution, I truncate the gaussian at +- 1 sigma.
	
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	num_freqs: Number of frequencies in image cube
	resolution: Array containing the resolution at all frequencies in pixel units 
	
	'''
	
	low_freq_ind_conv=0
	high_freq_ind_conv=1e5
	
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			freq_ind=y1*num_x+x1
			if low_freq_ind_conv>low_freq_ind[freq_ind]:
				low_freq_ind_conv=low_freq_ind[freq_ind]
			if high_freq_ind_conv<high_freq_ind[freq_ind]:
				high_freq_ind_conv=high_freq_ind[freq_ind]
	
	
	for y1 in range(num_y):
		for x1 in range(num_x):
			ind=y1*num_x*(num_params+1)+x1*(num_params+1)
			if fitted[ind]>0:
				smoothed_model_cube[y1,x1,:]=model[int(fitted[ind]),int(fitted[ind+1]),int(fitted[ind+2]),\
								int(fitted[ind+3]),int(fitted[ind+4]),:]
								
	for i in range(num_freqs):
		if i<low_freq_ind_conv or i>high_freq_ind_conv:
			continue
			
		res=resolution[i]
		sigma=res/(2*np.sqrt(2*np.log(2)))
		sigma_pix=int(sigma)+1
		truncate_sigma=1 ### I verified the source code. This takes +- truncate_sigma kernel
		low_indx_conv=max(low_indx-truncate_sigma*sigma_pix,0)
		low_indy_conv=max(low_indy-truncate_sigma*sigma_pix,0)
		high_indx_conv=min(high_indx+truncate_sigma*sigma_pix,num_x-1)
		high_indy_conv=min(high_indy+truncate_sigma*sigma_pix,num_y-1)
		
		smoothed_model_cube[low_indy_conv:high_indy_conv+1,\
					low_indx_conv:high_indx_conv+1,i]=\
									gaussian_filter(smoothed_model_cube[\
									low_indy_conv:high_indy_conv+1,\
									low_indx_conv:high_indx_conv+1,i],\
									sigma=sigma,mode='constant',cval=0.0,\
									truncate=truncate_sigma)
		
		
	return
	


def find_spatial_neighbours(cluster1,\
			     cluster2,\
			     nearest=True):
	'''
	This function takes one member of cluster2. Then it searches 
	neighbours within +-2 pixels and sees if those pixels are 
	located in cluster1. If yes, then it notes down the index
	number of that neighbour in the cluster1 list.
	
	cluster1: The list containing members of the biggest cluster
	cluster2: List containing memebers whose new parameters are
		  being searrched.
	nearest: Optional parameter. Decides whether we search for
		 neighbours between +-1 pix or +-2 pix 
	'''
	neighbours=[]
	for n,member in enumerate(cluster2):
		x0=member[0]
		y0=member[1]
		temp=[]
		k=0
		for i in range(-2,3):
			if nearest==True: 
				if i<-1 or  i>1:
					continue
			for j in range(-2,3):
				if nearest==True:
					if j<-1 or  j>1:
						continue
				k+=1
				try:
					ind=cluster1.index([x0+i,y0+j])
					temp.append(ind)
				except ValueError:
					pass
		neighbours.append(temp)
		del temp
	return neighbours		
	
def find_ind_combinations(x):
	'''
	This function produces this output
	input=[[1,2],[4,5,6]]
	output=[[1,4],[1,5],[1,6],[2,4],[2,5],[2,6]]
	
	x: List of lists
	'''
	lists=[]
	for elem in product(*x):
		lists.append(elem)
	return lists
	
def check_if_smoothness_condition_satisfied(cluster2, fitted,numx,numy,num_params):
	'''
	This function is used to reduce the computation time. While I am searching
	for all parameter combinations, if I find that the parameter difference from
	its neigbours is more than 1 array index, even though the neighbours is from
	the same cluster and I am finding the parameters for that point as well
	then I conlcude that this is not a smooth parameter combination.
	Hence there is no point in doing more computation. This function returns if 
	it is worth going through all the pain of computation, without doing the
	computation itself.
	
	cluster2: List containing members of smaller cluster
	fitted: Array containing current fit results
	numx,numy: Array size in X and Y coordinates
	num_params: Number of fitted parameters
	'''
	break_condition=False
	for n,member in enumerate(cluster2):
		x0=member[0]
		y0=member[1]
		ind0=y0*numx*(num_params+1)+x0*(num_params+1)
		for i in range(-1,2):
			for j in range(-1,2):
				try:
					ind=cluster2.index([x0+i,y0+j])
				except ValueError:
					continue
				x1=cluster2[ind][0]
				y1=cluster2[ind][1]
				ind1=y1*numx*(num_params+1)+x1*(num_params+1)
				
				for param in range(num_params):
					if abs(fitted[ind0+param]-fitted[ind1+param])>1:
						return False
	return True
						
										
def remove_big_clusters_image_comparison(clusters,\
					  cluster1,\
					  cluster2,\
					  spectral_cube,\
					  rms,\
					  sys_err,\
					  fitted, \
					  numx,\
					  numy,\
					  num_params,\
					  smooth_length,\
					  thresh,\
					  max_dist_parameter_space, \
					  model, \
					  low_indx,\
					  low_indy,\
					  high_indx,\
					  high_indy,\
					  min_params1,\
					  max_params1, \
					  resolution, \
					  low_freq_ind,\
					  upper_freq_ind,\
					  num_freqs,\
					  rms_thresh,\
					  param_lengths,\
					  smoothness_enforcer):
	'''
	This function tries to find new parameters for a small cluster
	close to the parameter values of the biggest cluster in that
	box.
	
	clusters: List of all clusters detected in the box
	cluster1: Members of the biggest cluster of the box
	cluster2: Members of the smaller cluster which would be analysed
		  in this call
	spectral_cube: Observed spectrum. Shape-num_timesx num_y x num_x x num_freqs
			At this moment num_times>1 has not been implemented
	rms: Image rms cube. Shape=  num_freqs.
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	fitted: array containing current fitted parameter values
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_length: smoothing length to be used
	thresh: Threshold used to detect discontinuity (not used probably)
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	low_indx,low_indy: the corodinate of the bottom-left corner of
	                   the box where the search should be done
	high_indx,high_indy: the corodinate of the top-right corner of
	                     the box where the search should be done
	min_params1, max_params1: List of the minimum and maximum allowed
				   allowed values of all parameters      
	resolution: resolution at all frequencies in pixel units
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	num_freqs: Number of frequencies in the image cube.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency               
	param_lengths: Lengths of the parameters in the supplied model cube
	'''				
	size_x=high_indx-low_indx+1
	size_y=high_indy-low_indy+1
	
	model_image_cube=np.zeros((numy,numx,num_freqs))
	observed_image_cube=spectral_cube[0,:,:,:]
	
	
	
	model_image_cube[:,:,:]=observed_image_cube
	
	create_smoothed_model_image(low_freq_ind,upper_freq_ind,numx, numy, num_params, num_freqs, resolution,model,fitted,\
					 model_image_cube, low_indx, low_indy, high_indx,high_indy)
	
	cfunc.get_image_chisquare(observed_image_cube,model_image_cube,rms,low_indx,low_indy,high_indx,high_indy,numx,numy,\
				num_params,fitted,sys_err, num_freqs,low_freq_ind, upper_freq_ind,rms_thresh)
	
	grad_chisquare_old=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy,numx,numy, num_params, \
							fitted,param_lengths,smoothness_enforcer)
	
	
	grad_chisquare_temp=grad_chisquare_old
	neighbours=find_spatial_neighbours(cluster1,cluster2,nearest=False)
	

	
			
	param_indices=[]
	for param in range(num_params):
		temp=[]
		for n,member in enumerate(cluster2):
			member_neighbour=neighbours[n]
			len_member_neighbour=len(member_neighbour)
			temp2=[]
			x0=member[0]
			y0=member[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)+param
			temp2.append(fitted[ind])
			if len_member_neighbour>0:
				for i in  range(len_member_neighbour):
					x0=cluster1[member_neighbour[i]][0]
					y0=cluster1[member_neighbour[i]][1]
					ind=y0*numx*(num_params+1)+x0*(num_params+1)+param
					temp2.append(fitted[ind])
			temp.append([i for i in temp2])
			del temp2
		param_indices.append(find_ind_combinations(temp))
		del temp		
		
	old_params=[]
	for n,member in enumerate(cluster2):
		x0=member[0]
		y0=member[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		for i in range(num_params+1):
			old_params.append(fitted[ind+i])	
			
	num_trials=len(param_indices[0])
	
	for i in range(num_trials):
		for n,member in enumerate(cluster2):
			x0=member[0]
			y0=member[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for j in range(num_params):
				fitted[ind+j]=param_indices[j][i][n]
		satisfies_smoothness_condition=check_if_smoothness_condition_satisfied(cluster2, fitted,numx,numy,num_params)
		if satisfies_smoothness_condition==False:
			continue
		create_smoothed_model_image(low_freq_ind,upper_freq_ind,numx, numy, num_params, num_freqs, resolution,model,\
						fitted, model_image_cube, low_indx, low_indy, high_indx,high_indy)
		cfunc.get_image_chisquare(observed_image_cube,model_image_cube,rms,low_indx,low_indy,high_indx,high_indy,numx,numy,\
				num_params,fitted,sys_err, num_freqs,low_freq_ind, upper_freq_ind,rms_thresh)
		grad_chisquare_new=cfunc.calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy,numx,numy, num_params, \
							fitted,param_lengths,smoothness_enforcer)
		if grad_chisquare_temp>grad_chisquare_new:
			del old_params
			old_params=[]
			for n,member in enumerate(cluster2):
				x0=member[0]
				y0=member[1]
				ind=y0*numx*(num_params+1)+x0*(num_params+1)
				for i in range(num_params+1):
					old_params.append(fitted[ind+i])	
			grad_chisquare_temp=grad_chisquare_new
	
	for n,member in enumerate(cluster2):
		x0=member[0]
		y0=member[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		for i in range(num_params+1):
			fitted[ind+i]=old_params[i]	
	return	
								
										
def smooth_param_maps_image_comparison(spectral_cube, \
					err_cube, \
					fitted, \
					param_val,\
					numx,\
					numy,\
					num_params,\
					smooth_lengths,\
					thresh,\
					max_dist_parameter_space, \
					model,\
					resolution, \
					param_lengths, \
					sys_err,\
					num_freqs,\
					low_freq_ind,\
					upper_freq_ind,\
					rms_thresh,\
					smoothness_enforcer,\
					max_iter=3):
	'''
	This function is a wrapper for the all the functions needed to find
	smooth parameter maps based on image smoothing according to frequency.
	The user calls this function and this calls others as needed.
	
	spectral_cube: Observed spectral cube, Shape: num_times, num_y, num_x, num_freqs
	err_cube: Image rms cube, shape: num_times, num_freqs
	fitted: the array containing the current fit results
	param_val: Actual values of the parameters
	numx,numy: Length along X and Y coordinates
	num_params: Number of fitted parameters
	smooth_lengths: The list of smoothing lengths to be used for finding
			 discontituiny
	thresh: Threshold used to detect discontinuity.
	max_dist_parameter_space: Maximum distance in parameter space. Used
	                          for cluster finding
	model: model cube supplied by the user
	resolution: Resolution at all frequencies in pixel units
	param_lengths: Lengths of the parameters in the supplied model cube
	sys_err: Systematic flux uncertainty; user input.
		    Added in quadrature (sys_error x flux at that frequency)
	
	num_freqs: Number of frequencies in the image cube.
	low_freq_ind,high_freq_ind: This contains the lowest and highest
	                            frequency index for which the spectrum
	                            can be described as a homogenous source spectrum.
	rms_thresh: Threshold in terms of image rms above which we can
	            treat that the source is not detected and we use the upper limit
	            as rms_thresh x rms at that frequency
	max_iter: Optional parameter. Maximum number of iterations for which the search
		  and removal process is repeated. If no points are changed in any
		  iteration search and removal process for that smoothing length
		  is stopped.
	'''
	j=0
	
	iter1=0
	max_iter=3
	
	err=np.ravel(err_cube[0,:])
	
	
	for smooth_length in smooth_lengths:
		while iter1<max_iter:
			subcubes=form_subcubes_with_gradients(numx,numy,num_params,fitted,smooth_length,param_lengths,smoothness_enforcer)
			grads=subcubes[:,2]
			sorted_indices=np.argsort(grads)[::-1]
			for sort_ind in sorted_indices:
				if subcubes[sort_ind,2]<0:
					continue
				x=int(subcubes[sort_ind,0])
				y=int(subcubes[sort_ind,1])
				
			
				
				low_indx=max(0,x-smooth_length//2)
				low_indy=max(0,y-smooth_length//2)

				high_indx=min(numx-1,x+smooth_length//2)
				high_indy=min(numy-1,y+smooth_length//2)
				
				
				if high_indx-low_indx+1<smooth_length or high_indy-low_indy+1<smooth_length:
					continue	
				
				clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space,param_lengths)
				
				if len(clusters)<=1:
					continue	
				tot_member=get_total_members(clusters)
				if tot_member<smooth_length**2:
					continue
				
				cluster_len=len(clusters)
				if cluster_len<=1:
					continue
					
					
				member_num=[]
				for cluster in clusters:
					member_num.append(len(cluster))	
				
				member_num=np.array(member_num)
				sorted_pos=np.argsort(member_num)[::-1]
				
				cluster1=clusters[sorted_pos[0]]
				
				
				len_cluster1=len(cluster1)
				
				min_params1=np.zeros(num_params,dtype=int)
				max_params1=np.zeros(num_params,dtype=int)
				for param in range(num_params):
					ind=[]
					for i in range(len_cluster1):
						x1=cluster1[i][0]
						y1=cluster1[i][1]
						ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
						ind.append(fitted[ind1])
					ind=np.array(ind)
					min_params1[param]=max(0,int(np.min(ind))-2)
					max_params1[param]=min(int(np.max(ind))+2,param_lengths[param]-1)
					del ind	
				
				
				for cluster_num in sorted_pos[1:]:
					if member_num[cluster_num]<0.5*member_num[sorted_pos[0]] and member_num[cluster_num]<8:
						
						remove_big_clusters_image_comparison(clusters,cluster1,clusters[cluster_num],\
										spectral_cube,err,sys_err,fitted, numx,numy,num_params,\
										smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,\
										high_indx,high_indy,min_params1,max_params1, resolution, low_freq_ind,\
										upper_freq_ind, num_freqs,rms_thresh,param_lengths,smoothness_enforcer)
						
				remove_overlapping_subcubes(subcubes,x,y,smooth_length,numy,numx)
				
			iter1+=1
		
			
	return
	
def main_func(xmin,\
	       ymin,\
	       xmax,\
	       ymax,\
	       lowest_freq,\
	       highest_freq,\
	       min_freq_num,\
	       spectrum,\
	       model,\
	       resolution,\
	       smooth_lengths,\
	       discontinuity_thresh=5,\
	       max_dist_parameter_space=4,\
	       sys_error=0.2,\
	       rms_thresh=3,\
	       smoothness_enforcer=0.05,\
	       outfile='outfile.hdf5'):
	       
	shape=model.model.shape
	spectrum_shape=spectrum.spectrum.shape
	model1=np.ravel(model.model)

	numx=spectrum_shape[2]
	numy=spectrum_shape[1]
	num_times=spectrum_shape[0]
	num_freqs=spectrum_shape[3]
	num_params=model.num_params

	param_lengths=np.ravel(np.zeros(len(model.param_names),dtype=np.intc))

	for i in range(num_params):
		param_lengths[i]=shape[i]
	max_len=np.max(param_lengths)

	spectrum_shape=np.shape(spectrum.spectrum)


	j=0
	tot_params=np.sum(param_lengths)

	param_vals=np.zeros(tot_params)

	for i in range(model.num_params):
		param_vals[j:j+param_lengths[i]]=model.param_vals[i]
		j=j+param_lengths[i]

	fitted=np.ravel(np.zeros(num_times*numy*numx*(num_params+1)))

	high_snr_freq_loc=np.ravel(np.zeros(num_times*numy*numx*num_freqs,dtype=np.intc))

	low_freq_ind=np.ravel(np.zeros(num_times*numy*numx,dtype=np.intc))
	upper_freq_ind=np.ravel(np.zeros(num_times*numy*numx,dtype=np.intc))
				
	
	spectrum1=np.ravel(spectrum.spectrum)
	error1=np.ravel(spectrum.error)
	model1=np.ravel(model.model)
	
	print ("doing pixel fit")
	cfunc.compute_min_chi_square(model1,spectrum1,error1,lowest_freq,\
		highest_freq,param_lengths,model.freqs,sys_error,rms_thresh,min_freq_num,\
		model.num_params, num_times,num_freqs,numy,numx,param_vals,high_snr_freq_loc,\
		fitted, low_freq_ind, upper_freq_ind)
		
	print ("removing discont")		
	remove_discont(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,num_params,\
			smooth_lengths,discontinuity_thresh,max_dist_parameter_space, model1,param_lengths,\
			sys_error,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,smoothness_enforcer)

	print ("Calling cluster remover")		
	smooth_param_maps(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,num_params,\
			smooth_lengths,discontinuity_thresh,max_dist_parameter_space, model1,param_lengths,\
			sys_error,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,smoothness_enforcer)
	
	print ("Doing image plane smoothing")	
	smooth_param_maps_image_comparison(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,\
					num_params,smooth_lengths,discontinuity_thresh,max_dist_parameter_space,\
					model.model,resolution,param_lengths,sys_error,num_freqs,low_freq_ind,\
					upper_freq_ind,rms_thresh,smoothness_enforcer)
	
				
	param_maps=np.zeros((num_times,numy,numx,num_params))
	chi_map=np.zeros((num_times,numy,numx))

	for t in range(num_times):
		for y1 in range(numy):
			for x1 in range(numx):
				for m in range(num_params):
					ind=t*numy*numx*(num_params+1)+y1*numx*(num_params+1)+x1*(num_params+1)+m
					param_maps[t,y1,x1,m]=model.param_vals[m][int(fitted[ind])]
				chi_map[t,y1,x1]=fitted[int(ind+1)]

	param_names=model.param_names

	hf=h5py.File(outfile,'w')
	hf.attrs['xmin']=xmin
	hf.attrs['ymin']=ymin
	hf.attrs['xmax']=xmax
	hf.attrs['ymax']=ymax
	hf.attrs['lower_freq']=lowest_freq
	hf.attrs['upper_freq']=highest_freq
	hf.attrs['rms_thresh']=rms_thresh
	hf.attrs['sys_error']=sys_error
	hf.attrs['min_freq_num']=min_freq_num
	hf.attrs['discontinuity_thresh']=discontinuity_thresh
	hf.attrs['max_dist_parameter_space']=max_dist_parameter_space
	hf.attrs['smoothness_enforcer']=smoothness_enforcer

	hf.create_dataset('low_freq_ind',data=low_freq_ind)
	hf.create_dataset('upper_freq_ind',data=upper_freq_ind)
	hf.create_dataset('smooth_lengths',data=np.array(smooth_lengths))

	for n,key in enumerate(param_names):
		hf.create_dataset(key,data=param_maps[:,:,:,n])
	hf.create_dataset('chi_sq',data=chi_map[:,:,:])
	hf.close()
	
	return
