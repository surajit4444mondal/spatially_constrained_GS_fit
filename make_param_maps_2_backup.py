import numpy as np
import matplotlib.pyplot as plt
import time
#from get_model import Model
from spectrum import Spectrum
from gs_minimizer import GS_minimizer
import calculate_chi_square as chi_mod
import h5py

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

def calc_gradx(x0,y0,param,param_val,fitted, numx,numy,num_params):
	
	x1=x0-1
	y1=y0
	ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	grad1=0
	if x1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		#grad1=(param_val[param][int(fitted[ind])]-param_val[param][int(fitted[ind1])])
		grad1=(fitted[ind]-fitted[ind1])
		
	x2=x0+1
	y2=y0
	ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	grad2=0
	if x2>numx-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		#grad2=(param_val[param][int(fitted[ind2])]-param_val[param][int(fitted[ind])])
		grad2=(fitted[ind2]-fitted[ind])
		
	grad=np.sqrt(grad1**2+grad2**2)
	return grad
	
def calc_grady(x0,y0,param,param_val,fitted, numx,numy,num_params):
	
	x1=x0
	y1=y0-1
	ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	grad1=0
	if y1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		#grad1=(param_val[param][int(fitted[ind])]-param_val[param][int(fitted[ind1])])
		grad1=(fitted[ind]-fitted[ind1])
		
	x2=x0
	y2=y0+1
	ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	grad2=0
	if y2>numy-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		#grad2=(param_val[param][int(fitted[ind2])]-param_val[param][int(fitted[ind])])/2
		grad2=(fitted[ind2]-fitted[ind])
		
	grad=np.sqrt(grad1**2+grad2**2)
	return grad	

def calc_gradient(x0,y0,param_val,fitted, num_x,num_y,num_params):
	ind=y0*num_x*(num_params+1)+x0*(num_params+1)+num_params
	if fitted[ind]<0:
		return 0

	grad=0.0	
	for param in range(num_params):
		gradx=calc_gradx(x0,y0,param,param_val,fitted, num_x,num_y,num_params)
		grady=calc_grady(x0,y0,param,param_val,fitted, num_x,num_y,num_params)
		grad=grad+gradx**2+grady**2
	return grad
		
def calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy,numx,numy, num_params, param_val,fitted):
	chi_square=0
	for y1 in range(low_indy, high_indy):
		for x1 in range(low_indx,high_indx):
			ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind]>0:
				chi_square=chi_square+fitted[ind]+smoothness_enforcer*\
						calc_gradient(x1,y1,param_val,fitted, numx,numy,num_params)
						
	return chi_square


def get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space):
	
	cluster_list=[]
	
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
						dist=dist+(fitted[ind1]-fitted[ind0])**2
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
	
def calc_chi_square(spectrum, rms, sys_error, model_spectrum):
	chi_square=0
	low_ind=2
	high_ind=40
	rms_thresh=5
	for freq_ind in range(low_ind,high_ind):
		if spectrum[freq_ind]>rms_thresh*rms[freq_ind]:
			error=np.sqrt(rms[freq_ind]**2+(sys_error*spectrum[freq_ind])**2)
			chi_square+=(spectrum[freq_ind]-model_spectrum[freq_ind])**2/error**2
		else:
			if model_spectrum[freq_ind]<rms_thresh*rms[freq_ind]:
				chi_square+=0.0
			else:
				chi_square+=100.0
	return chi_square
	
def find_new_params(points_to_remove,fitted,model,spectrum,sys_error,rms,param_val,numx,numy,num_params, low_indx,low_indy,high_indx,high_indy):
	new_params=[]
	for m,point in enumerate(points_to_remove):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		new_params.append([])
		for i in range(ind,ind+num_params+1):
			new_params[m].append(fitted[i])
		inds=[]
		for i in range(-1,2):
			for j in range(-1,2):
				x1=x0+j
				y1=y0+i
				if [x1,y1] not in points_to_remove:
					ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
					if fitted[ind]>0:
						inds.append(fitted[ind-num_params:ind].tolist())
		
		grad_chi_square=calc_grad_chisquare(max(low_indx, x0-2),max(low_indy,y0-2),min(high_indx,x0+2),min(high_indy,y0+2), numx,numy,num_params, param_val,fitted)	
		for n,param_ind in enumerate(inds):
			model1=model
			for k in range(num_params):
				model1=model1[int(param_ind[k])]
				fitted[y0*numx*(num_params+1)+x0*(num_params+1)+k]=param_ind[k]
			chi_square=calc_chi_square(spectrum[:,y0-low_indy,x0-low_indx],rms,sys_error,model1)
			fitted[y0*numx*(num_params+1)+x0*(num_params+1)+num_params]=chi_square
			grad_chi_square_temp=calc_grad_chisquare(max(low_indx, x0-2),max(low_indy,y0-2),min(high_indx,x0+2),min(high_indy,y0+2), numx,numy,num_params, param_val,fitted)		
			if grad_chi_square_temp<grad_chi_square:
				grad_chi_square=grad_chi_square_temp
				for param1 in range(num_params):
					new_params[m][param1]=param_ind[param1]
				new_params[m][num_params]=chi_square
		for k in range(num_params+1):
			fitted[y0*numx*(num_params+1)+x0*(num_params+1)+k]=new_params[m][k]
	return new_params
					
def remove_discont(spectral_cube, err_cube, fitted, param_val,numx,numy,num_params,search_length,thresh,max_dist_parameter_space, model):

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
		
		spectrum=spectral_cube[0,:,low_indy:high_indy+1,low_indx:high_indx+1]
		rms=err_cube[0,:]
		sys_err=0.2
		
			
		clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space)
		
		points_to_remove=get_points_to_remove(clusters)
		
		remove_all_points(points_to_remove,spectrum,rms,sys_err, fitted, param_val,numx,numy,num_params,\
					search_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy)
	
		for discont1 in discont_list[point_num+1:]:
			if discont1[0]>=low_indx and discont1[0]<=high_indx	and discont1[1]>=low_indy and discont1[1]<=high_indy:
				discont1[0]=-1
				discont1[1]=-1		
	return		

def remove_all_points(points_to_remove,spectrum,rms,sys_err, fitted, param_val,numx,numy,num_params,\
					search_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy):

	old_params=[]
	for m,point in enumerate(points_to_remove):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		old_params.append([])
		for i in range(ind,ind+num_params+1):
			old_params[m].append(fitted[i])

	grad_chi_square_old=calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, param_val,fitted)	
	
	new_params=find_new_params(points_to_remove,fitted,model,spectrum,sys_err,rms,param_val,numx,numy,num_params, low_indx,low_indy,high_indx,high_indy)
	
	for m,point in enumerate(points_to_remove):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		for k,val in enumerate(new_params[m]):
			fitted[ind+k]=val
	grad_chi_square_new=calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, param_val,fitted)	
	
	if grad_chi_square_new>grad_chi_square_old:
		for m,point in enumerate(points_to_remove):
			x0=point[0]
			y0=point[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for k,val in enumerate(old_params[m]):
				fitted[ind+k]=val	
	return	
	
def get_new_param_inds(spectrum,rms,model,min_param1,max_param1):
	model_shape=model.shape
	chisq=1e9
	params=[-1,-1,-1,-1,-1,-1]
	for i in range(min_param1[0],max_param1[0]+1):
		for j in range(min_param1[1],max_param1[1]+1):
			for k in range(min_param1[2],max_param1[2]+1):
				for l in range(min_param1[3],max_param1[3]+1):
					for m in range(min_param1[4],max_param1[4]+1):
						min_freq_ind=2
						max_freq_ind=40
						model_spec=model[i,j,k,l,m,:]
						mid_ind=(min_freq_ind+max_freq_ind)//2
						ratio=spectrum[mid_ind]/model_spec[mid_ind]
						if ratio>2 or ratio<0.5:
							continue
						chisq_temp=calc_chi_square(spectrum, rms,0.2, model_spec)
						if chisq_temp<chisq:
							chisq=chisq_temp
							params[0]=i
							params[1]=j
							params[2]=k
							params[3]=l
							params[4]=m
							params[5]=chisq
	return params
	
def remove_big_clusters(clusters,cluster1,cluster2,spectral_cube,err_cube,sys_err,fitted, param_val,numx,numy,num_params,\
								smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy,\
								min_params1,max_params1):
	'''
	cluster_len=len(clusters)
	if cluster_len<=1:
		return
		
	member_num=[]
	for cluster in clusters:
		member_num.append(len(cluster))	
	
	member_num=np.array(member_num)
	sorted_pos=np.argsort(member_num)
	
	if len(clusters[sorted_pos[cluster_len-1]])>len(clusters[sorted_pos[cluster_len-2]]):
		cluster1=clusters[sorted_pos[cluster_len-1]]
		cluster2=clusters[sorted_pos[cluster_len-2]]
	else:
		cluster2=clusters[sorted_pos[cluster_len-1]]
		cluster1=clusters[sorted_pos[cluster_len-2]]
	
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
		min_params1[param]=int(np.min(ind))-2
		max_params1[param]=int(np.max(ind))+2
		del ind	
	'''
	old_params=[]
	for m,point in enumerate(cluster2):
		x0=point[0]
		y0=point[1]
		ind=y0*numx*(num_params+1)+x0*(num_params+1)
		old_params.append([])
		for i in range(ind,ind+num_params+1):
			old_params[m].append(fitted[i])	
			
	grad_chi_square_old=calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, param_val,fitted)
	
	rms=err_cube[0,:]

	for m,point in enumerate(cluster2):
		x0=point[0]
		y0=point[1]
		spectrum=spectral_cube[0,:,y0,x0]
		new_params_temp=get_new_param_inds(spectrum,rms,model,min_params1,max_params1)	
		if new_params_temp[0]>0:
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for i in range(ind,ind+num_params+1):
				fitted[i]=new_params_temp[i-ind]
				
		
	grad_chi_square_new=calc_grad_chisquare(low_indx,low_indy,high_indx,high_indy, numx,numy,num_params, param_val,fitted)
	
	print (grad_chi_square_old,grad_chi_square_new)
	
	if grad_chi_square_old<grad_chi_square_new:
		for m,point in enumerate(cluster2):
			x0=point[0]
			y0=point[1]
			ind=y0*numx*(num_params+1)+x0*(num_params+1)
			for k,val in enumerate(old_params[m]):
				fitted[ind+k]=val
	
	
def get_total_members(clusters):
	member_num=0
	for cluster in clusters:
		member_num+=len(cluster)
	return member_num

def smooth_param_maps(spectral_cube, err_cube, fitted, param_val,numx,numy,num_params,smooth_lengths,thresh,max_dist_parameter_space, model):
	j=0
	model_shape=model.shape

	for smooth_length in smooth_lengths:
		for y in range(0,numy,smooth_length):
			for x in range(0,numx,smooth_length):
				low_indx=max(0,x-smooth_length//2)
				low_indy=max(0,y-smooth_length//2)

				high_indx=min(numx-1,x+smooth_length//2)
				high_indy=min(numy-1,y+smooth_length//2)
				
				
				if high_indx-low_indx+1<smooth_length or high_indy-low_indy+1<smooth_length:
					continue	
				
				clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params,max_dist_parameter_space*1.5)
				if len(clusters)<=1:
					continue	
				tot_member=get_total_members(clusters)
				if tot_member<smooth_length**2:
					continue
				low_indx=max(0,x-search_length//2)
				low_indy=max(0,y-search_length//2)

				high_indx=min(numx-1,x+search_length//2)
				high_indy=min(numy-1,y+search_length//2)
				
				if high_indx-low_indx<search_length or high_indy-low_indy<search_length:
					continue
				
				spectrum=spectral_cube[0,:,low_indy:high_indy+1,low_indx:high_indx+1]
				rms=err_cube[0,:]
				sys_err=0.2
				points_to_remove=get_points_to_remove(clusters)
			
				remove_all_points(points_to_remove,spectrum,rms,sys_err, fitted, param_val,numx,numy,num_params,\
							smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy)
							
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
					max_params1[param]=min(int(np.max(ind))+2,model_shape[param]-1)
					del ind	
				
				for cluster_num in sorted_pos[1:2]:					
					remove_big_clusters(clusters,clusters[sorted_pos[0]],clusters[cluster_num],spectral_cube,err_cube,sys_err,fitted, param_val,numx,numy,num_params,\
										smooth_length,thresh,max_dist_parameter_space, model, low_indx,low_indy,high_indx,high_indy,min_params1,max_params1)
			
			
		
		
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
outfile='model_paramaters_'+time_slice+'.hdf5'
lower_freq=2
upper_freq=14

xmin=519
ymin=482
xmax=576
ymax=558
lowest_freq=2
highest_freq=15
sys_error=0.2
rms_thresh=3.0
min_freq_num=10
search_length=10
smooth_lengths=[5,9]
discontinuity_thresh=5.0
spatial_smooth=0.05
temporal_smooth=0.0
frac_tol=0.1
max_iter=10
max_dist_parameter_space=4


spectrum_files=['/home/surajit/Downloads/20210507/eovsa_data/time_'+time_slice+'_scaled.fits']

model=Model(model_file)
spectrum=Spectrum(spectrum_files,xmin,ymin,xmax,ymax,lowest_freq,highest_freq)
spectrum.read_map()

hf=h5py.File('test_cython_code.hdf5')
fitted=np.array(hf['fitted'])
hf.close()

spectrum_shape=spectrum.spectrum.shape


j=0

ind=15*spectrum_shape[3]*6+17*6
print (fitted[ind:ind+6])
remove_discont(spectrum.spectrum, spectrum.error, fitted, model.param_vals,spectrum_shape[3],spectrum_shape[2],5,search_length,discontinuity_thresh,max_dist_parameter_space, model.model)
print ("Calling smooth_param_maps")
smooth_param_maps(spectrum.spectrum, spectrum.error, fitted, model.param_vals,spectrum_shape[3],spectrum_shape[2],5,smooth_lengths,discontinuity_thresh,max_dist_parameter_space*2, model.model)

f1=fitted.reshape((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3],model.num_params+1))
print(f1[0,15,17,:])


param_maps=np.zeros((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3],model.num_params))
chi_map=np.zeros((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3]))

for t in range(spectrum_shape[0]):
	for y1 in range(spectrum_shape[2]):
		for x1 in range(spectrum_shape[3]):
			for m in range(model.num_params):
				param_maps[t,y1,x1,m]=model.param_vals[m][int(f1[t,y1,x1,m])]
			chi_map[t,y1,x1]=f1[t,y1,x1,model.num_params]

param_names=model.param_names

hf=h5py.File("fitted_param_maps_"+time_slice+"_smoothness_enforced_loop_over_cluster_python.hdf5",'w')
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

for n,key in enumerate(param_names):
	hf.create_dataset(key,data=param_maps[:,:,:,n])
hf.create_dataset('chi_sq',data=chi_map[:,:,:])
hf.close()


