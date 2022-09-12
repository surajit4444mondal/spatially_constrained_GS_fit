import numpy as np
import matplotlib.pyplot as plt
import time
#from get_model import Model
from spectrum import Spectrum
import cython_functions_for_fast_computation as cfunc
import h5py
from scipy.ndimage import gaussian_filter
from itertools import product
from astropy.convolution import convolve,Gaussian1DKernel

smoothness_enforcer=0.05

def check_at_boundary(x0,y0,numx,numy,num_params,fitted):
	x=[x0-1,x0+1,x0,x0,x0-1,x0+1,x0-1,x0-1]
	y=[y0,y0,y0-1,y0+1,y0+1,y0+1,y0-1,y0+1]
	
	for x1,y1 in zip(x,y):
		ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
		if fitted[ind]<0:
			return 1
	
	return 0


def detect_discont(fitted, param_val,x,y,numx,numy,num_params,search_length,thresh):
	
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
		
		
def list_discont(fitted, param_val,numx,numy,num_params,search_length,thresh):
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




def get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space,param_lengths):
	
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
	
	member_num=[]
	for cluster in cluster_list:
		member_num.append(len(cluster))
	
	min_member=min(member_num)
	max_member=4
	
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
	
	
def find_new_params(points_to_remove,fitted,model,spectrum,sys_error,rms,param_val,numx,numy,num_params,\
			 low_indx,low_indy,high_indx,high_indy,param_lengths,num_freqs,low_freq_ind,upper_freq_ind,\
			 rms_thresh):
	new_params=[]
	points_changed=0
	changed=False
	for m,point in enumerate(points_to_remove):
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
			 max_iter=5):

	
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
							param_lengths,num_freqs,low_freq_ind,upper_freq_ind, rms_thresh)
				
				for discont1 in discont_list[point_num+1:]:
					if discont1[0]>=low_indx and discont1[0]<=high_indx	and discont1[1]>=low_indy and discont1[1]<=high_indy:
						discont1[0]=-1
						discont1[1]=-1	
					
			iter1+=1
			if points_changed==0:
				break
		
	return		

def remove_all_points(points_to_remove,spectrum,rms,sys_err, fitted, param_val,numx,numy,num_params,\
					search_length,thresh,max_dist_parameter_space, model, low_indx,\
					low_indy,high_indx,high_indy,param_lengths,num_freqs,\
					low_freq_ind,upper_freq_ind,rms_thresh):

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
							 upper_freq_ind, rms_thresh)
	
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
	
def get_new_param_inds(spectrum,rms,model,min_params1,max_params1,param_lengths, new_params_temp,low_freq_ind,\
							upper_freq_ind,rms_thresh,num_params, num_freqs,sys_err):

	possible_param_indices=[None]*num_params
	for i in range(num_params):
		possible_param_indices[i]=np.arange(min_params1[i],max_params1[i]+1)
	
	
	chisq=np.array([1e9])
	for i in product(*possible_param_indices):
		param_ind=np.array(i,dtype=np.intc)
		cfunc.get_new_param_inds_general(spectrum,rms,model,param_ind,param_lengths, new_params_temp,low_freq_ind,\
							upper_freq_ind,rms_thresh,num_params, num_freqs,sys_err,chisq)
	return
	
def remove_big_clusters(clusters,cluster1,cluster2,spectral_cube,err_cube,sys_err,fitted, param_val,numx,numy,num_params,\
								smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy,\
								min_params1,max_params1,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,param_lengths):
	
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
	
	print (grad_chi_square_old,grad_chi_square_new)
	
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
	member_num=0
	for cluster in clusters:
		member_num+=len(cluster)
	return member_num
	
def form_subcubes_with_gradients(numx,numy,num_params,fitted,smooth_length,param_lengths):
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

def smooth_param_maps(spectral_cube, err_cube, fitted,param_val,numx,numy,num_params,\
			smooth_lengths,thresh,max_dist_parameter_space, model,param_lengths,\
			sys_err,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,max_iter=5):
	j=0
	
	
	
	#subcubes=form_subcubes_with_gradients(numx,numy,num_params,param_val,fitted,smooth_lengths[0],param_lengths)
	
	
	#grads=subcubes[:,2]
	#sorted_indices=np.argsort(grads)[::-1]
	
	for smooth_length in smooth_lengths:
		iter1=0
		changed_points=0
		while iter1<max_iter:
			subcubes=form_subcubes_with_gradients(numx,numy,num_params,fitted,smooth_length,param_lengths)
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
				
				
							
				print (x,y)
				
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
											num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,param_lengths)
					if changed==True:
						changed_points+=1
				remove_overlapping_subcubes(subcubes,x,y,smooth_lengths[0],numy,numx)
			if changed_points==0:
				break
			iter1+=1

def create_smoothed_model_image(low_freq_ind,high_freq_ind,num_x, num_y, num_params, num_freqs, resolution,\
					model,fitted, smoothed_model_cube, low_indx, low_indy, high_indx,high_indy):

	low_freq_ind_conv=0
	high_freq_ind_conv=1e5
	
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indy+1):
			freq_ind=y1*num_freqs+x1
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
		truncate_sigma=3
		low_indx_conv=max(low_indx-truncate_sigma*sigma_pix,0)
		low_indy_conv=max(low_indy-truncate_sigma*sigma_pix,0)
		high_indx_conv=min(high_indx+truncate_sigma*sigma_pix,num_x-1)
		high_indy_conv=min(high_indy+truncate_sigma*sigma_pix,num_y-1)
		
		#kernel=Gaussian2DKernel(sigma_pix,sigma_pix,x_size=2*truncate_sigma*sigma_pix+1,\
		#			y_size=2*truncate_sigma*sigma_pix+1)
		
		smoothed_model_cube[low_indy_conv:high_indy_conv+1,\
					low_indx_conv:high_indx_conv+1,i]=\
									gaussian_filter(smoothed_model_cube[\
									low_indy_conv:high_indy_conv+1,\
									low_indx_conv:high_indx_conv+1,i],\
									sigma=sigma,mode='constant',cval=0.0,\
									truncate=truncate_sigma)
		
		
	return
	


def find_spatial_neighbours(cluster1,cluster2,nearest=True):
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
	lists=[]
	for elem in product(*x):
		lists.append(elem)
	return lists
	
def check_if_smoothness_condition_satisfied(cluster2, fitted,numx,numy,num_params):
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
						
										
def remove_big_clusters_image_comparison(clusters,cluster1,cluster2,\
					spectral_cube,rms,sys_err,fitted, numx,numy,num_params,\
					smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,\
					high_indx,high_indy,min_params1,max_params1, resolution, low_freq_ind,\
					upper_freq_ind,num_freqs,rms_thresh):
					
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
	#print (param_indices[0])	
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
	print (grad_chisquare_old,grad_chisquare_temp)	
	return	
								
										
def smooth_param_maps_image_comparison(spectral_cube, err_cube, fitted, param_val,numx,numy,num_params,\
					smooth_lengths,thresh,max_dist_parameter_space, model,resolution, \
					param_lengths, sys_err,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh,\
					max_iter=3):
	j=0
	
	iter1=0
	max_iter=3
	
	err=np.ravel(err_cube[0,:])
	
	
	for smooth_length in smooth_lengths:
		while iter1<max_iter:
			subcubes=form_subcubes_with_gradients(numx,numy,num_params,fitted,smooth_length,param_lengths)
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
				
				print (x,y)
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
					
				if cluster_len>6:
					clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space*1.5,param_lengths)
					cluster_len=len(clusters)
					
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
										upper_freq_ind, num_freqs,rms_thresh)
				remove_overlapping_subcubes(subcubes,x,y,smooth_length,numy,numx)
			iter1+=1
			

	

		
class Model:
	'''
	The model is such that the last axes (in the cython code this is the fastest traversing axes similar to
	C style) is the spectrum for the corresponding model vals.
	'''
	def __init__(self,filename):
		hf=h5py.File(filename)
		keys=['Bx100G','delta','log_nnth','log_nth','theta','spectrum','freqs']#hf.keys()
		num_params=len(keys)-2  ### 1 is for spectrum, 1 is for freqs
		self.num_params=num_params
		self.model=np.array(hf['spectrum'])
		self.freqs=np.array(hf['freqs'])
		params=[]
		param_names=[]
		for k in keys:
			if k!='spectrum' and k!='freqs':
				params.append(np.array(hf[k]))
				param_names.append(k)
		self.param_vals=params
		self.param_names=param_names
		hf.close()

time_slice='190200_190210'
model_file='/home/surajit/Downloads/20210507/eovsa_data/model_gs_spectra.hdf5'

xmin=519
ymin=482
xmax=576
ymax=558
lowest_freq=2
highest_freq=15
sys_error=0.2
rms_thresh=3.0
min_freq_num=35
smooth_lengths=[4,9]
discontinuity_thresh=5.0
spatial_smooth=0.05
temporal_smooth=0.0
frac_tol=0.1
max_iter=10
max_dist_parameter_space=4
cell=0.5  ###arcsec

freqs=np.loadtxt("/home/surajit/Downloads/20210507/eovsa_data/ctrl_freqs.txt")
num_freqs=np.size(freqs)

resolution=np.zeros(num_freqs)

for i in range(num_freqs):
	resolution[i]=(max(60/(freqs[i]*1e-3),3))/cell 


spectrum_files=['/home/surajit/Downloads/20210507/eovsa_data/time_'+time_slice+'_scaled.fits']

model=Model(model_file)
spectrum=Spectrum(spectrum_files,xmin,ymin,xmax,ymax,lowest_freq,highest_freq)
spectrum.read_map()

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
model1=np.ravel(model.model)

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


#cfunc.compute_min_chi_square(model1,spectrum1,error1,lowest_freq,\
#		highest_freq,param_lengths,model.freqs,sys_error,rms_thresh,min_freq_num,\
#		model.num_params, num_times,num_freqs,numy,numx,param_vals,high_snr_freq_loc,\
#		fitted, low_freq_ind, upper_freq_ind)
		
hf=h5py.File("python_cython_comb_test.hdf5")
#fitted=np.array(hf['fitted'])
#np.save("python_cython_comb_test.npy",fitted)
fitted=np.load("python_cython_comb_test.npy")
low_freq_ind=np.array(hf['low_freq_ind'])
upper_freq_ind=np.array(hf['upper_freq_ind'])
hf.close()

print ("removing discont")		
remove_discont(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,num_params,\
		smooth_lengths,discontinuity_thresh,max_dist_parameter_space, model1,param_lengths,\
		sys_error,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh)
		
smooth_param_maps(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,num_params,\
		smooth_lengths,discontinuity_thresh,max_dist_parameter_space, model1,param_lengths,\
		sys_error,num_freqs,low_freq_ind,upper_freq_ind,rms_thresh)
		
#np.save("python_cython_comb_cluster_removal_test.npy",fitted)
fitted=np.load("python_cython_comb_cluster_removal_test.npy")		
smooth_param_maps_image_comparison(spectrum.spectrum, spectrum.error, fitted, model.param_vals,numx,numy,\
				num_params,smooth_lengths,discontinuity_thresh,max_dist_parameter_space,\
				model.model,resolution,param_lengths,sys_error,num_freqs,low_freq_ind,\
				upper_freq_ind,rms_thresh)
				

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

hf=h5py.File("python_cython_comb_discont_removal_cluster_removal_test1.hdf5",'w')
hf.attrs['xmin']=xmin
hf.attrs['ymin']=ymin
hf.attrs['xmax']=xmax
hf.attrs['ymax']=ymax
hf.attrs['timeslice']=time_slice
hf.attrs['lower_freq']=lowest_freq
hf.attrs['upper_freq']=highest_freq
hf.attrs['rms_thresh']=rms_thresh
hf.attrs['sys_error']=sys_error
hf.attrs['min_freq_num']=min_freq_num

#hf.create_dataset('fitted',data=fitted)
hf.create_dataset('low_freq_ind',data=low_freq_ind)
hf.create_dataset('upper_freq_ind',data=upper_freq_ind)

for n,key in enumerate(param_names):
	hf.create_dataset(key,data=param_maps[:,:,:,n])
hf.create_dataset('chi_sq',data=chi_map[:,:,:])
hf.close()



