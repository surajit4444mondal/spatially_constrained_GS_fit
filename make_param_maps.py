import numpy as np
import matplotlib.pyplot as plt
import time
#from get_model import Model
from spectrum import Spectrum
from gs_minimizer import GS_minimizer
import calculate_chi_square as chi_mod
import h5py

smoothness_enforcer=0.05


def detect_discont(fitted, param_val,x,y,numx,numy,num_params,search_length,thresh):
	low_indx=max(0,x-search_length//2)
	low_indy=max(0,y-search_length//2)

	high_indx=min(numx-1,x+search_length//2)
	high_indy=min(numy-1,y+search_length//2)
	
	if high_indx-low_indx<absolute_length or high_indy-low_indy<absolute_length:
		return 0
	
	for_median=np.zeros(int(search_length**2))
	
	
	for param in range(num_params):
		j=0
		for_median[:]=0
		for y1 in range(low_indy,high_indy+1):
			for x1 in range(low_indx,high_indx+1):
				ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
				if fitted[ind]<0:
					continue
				ind=y1*numx*(num_params+1)+x1*(num_params+1)+param
				for_median[j]=int(fitted[ind])
		median=np.median(for_median)
		mad=np.median(np.absolute(for_median-median))
		ind_center=y1*numx*(num_params+1)+x1*(num_params+1)+param
		
		if mad<1e-4 and median<1e-4:
			return 0
		elif mad<1e-4 and abs(median-fitted[ind_center])>1e-4:
			return 1
		elif absolute(median-fitted[ind_center])>thresh*mad:
			return 1
		
		
def list_discont(fitted, param_val,numx,numy,num_params,search_length,thresh):
	discont=[]
	for x in range(numx):
		for y in range(numy):
			ind5=y*numx*(num_params+1)+x*(num_params+1)+num_params
			if fitted[ind5]<0:
				continue
			discont_found=detect_discont(fitted, param_val,x,y,numx,numy,num_params,search_length,thresh)
			if discont_found==1:
				discont.append([x,y])

def calc_gradx(x0,y0,param,param_val,fitted, num_x,num_y,num_params):
	
	x1=x0-1
	y1=y0
	ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	grad1=0
	if x1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		grad1=(fitted[ind]-fitted[ind1])/2
		
	x2=x0+1
	y2=y0
	ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	grad2=0
	if x2>num_x-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		grad2=(fitted[ind2]-fitted[ind])/2
		
	grad=np.sqrt(grad1**2+grad2**2)
	return grad
	
def calc_grady(x0,y0,param,param_val,fitted, num_x,num_y,num_params):
	
	x1=x0
	y1=y0-1
	ind1=y1*numx*(num_params+1)+x1*(num_params+1)+param
	ind=y0*numx*(num_params+1)+x0*(num_params+1)+param

	
	grad1=0
	if y1<0:
		grad1=0
	elif fitted[ind1]>0 and fitted[ind]>0:
		grad1=(param_val[param][int(fitted[ind])]-param_val[param][int(fitted[ind1])])/2
		
	x2=x0
	y2=y0+1
	ind2=y2*numx*(num_params+1)+x2*(num_params+1)+param
	grad2=0
	if y2>num_x-1:
		grad2=0
	if fitted[ind2]>0 and fitted[ind]>0:
		grad2=(param_val[param][int(fitted[ind2])]-param_val[param][int(fitted[ind])])/2
		
	grad=np.sqrt(grad1**2+grad2**2)
	return grad	

def calc_gradient(x0,y0,param_val,fitted, num_x,num_y,num_params):

	ind=y0*numx*(num_params+1)+x0*(num_params+1)+num_params
	if fitted[ind]<0:
		return 0

	grad=0.0	
	for param in range(num_params):
		gradx=calc_gradx(x0,y0,param,param_val,fitted, num_x,num_y,num_params)
		grady=calc_grady(x0,y0,param,param_val,fitted, num_x,num_y,num_params)
		grad=grad+gradx**2+grad_y**2
	return grad
		
def calc_grad_chisquare(spectrum, error, low_indx,low_indy,high_indx,high_indy, num_params, param_val,fitted):
	chi_square=0
	for y1 in range(low_indy, high_indy):
		for x1 in range(low_indx,high_indx):
			ind=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind]>0:
				chi_square=chi_square+fitted[ind]+smoothness_enforcer*\
						calc_gradient(x1,y1,param_val,fitted, num_x,num_y,num_params)
						
	return chi_square


def get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numx,num_params):
	
	cluster_list=[]
	
	for y1 in range(low_indy,high_indy+1):
		for x1 in range(low_indx,high_indx+1):
			ind0=y1*numx*(num_params+1)+x1*(num_params+1)+num_params
			if fitted[ind0]<0:
				continue
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
					if dist<min_dist:
						cluster_member=True
						clusters.append(n)
						break
			if len(clusters)==0:
				cluster_list.append([x1,y1])
			elif len(clusters)==1:
				cluster_list[clusters[0]].append([x1,y1])
			else:
				cluster_list[clusters[0]].append([x1,y1])
				for m in cluster_list[clusters[1]]:
					cluster_list[cluster[0]].append(m)
				del cluster_list[clusters[1]]
		
	return cluster_list
				
def remove_discont(spectral_cube, err_cube, fitted, param_val,numx,numy,num_params,search_length,thresh):

	discont_list=list_discont(fitted, param_val,numx,numy,num_params,search_length,thresh)
	
	for discont in discont_list:
		x=discont[0]
		y=discont[1]
		low_indx=max(0,x-search_length//2)
		low_indy=max(0,y-search_length//2)

		high_indx=min(numx-1,x+search_length//2)
		high_indy=min(numy-1,y+search_length//2)
		
		if high_indx-low_indx<search_length or high_indy-low_indy<search_length:
			continue
		
		spectrum=spectral_cube[0,:,low_indy:high_indy+1,low_indx:high_indx+1]
		rms=err_cube[0,:]
		sys_err=0.2
		error=np.zeros(spectrum.shape)
		
		i=0
		for y1 in range(low_indy,high_indy+1):
			j=0
			for x1 in range(low_indx,high_indx+1):
				error[:,i,j]=np.sqrt(rms**2+(sys_err*spectrum[:,i,j])**2)
				
		clusters=get_clusters(fitted,low_indx,low_indy, high_indx,high_indy,numx,numy,num_params)
		
		
		
		do_gradient_minimisation(spectrum,error,fitted, low_indx,low_indy,high_indx,high_indy,num_params,param_val)
			
			
			
			
			
			
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

'''
xmin=550
ymin=520
xmax=551
ymax=521
lowest_freq=2
highest_freq=15
sys_error=0.2
rms_thresh=3.0
min_freq_num=10
'''
xmin=519
ymin=482
xmax=576
ymax=558
lowest_freq=2
highest_freq=15
sys_error=0.2
rms_thresh=3.0
min_freq_num=10


spectrum_files=['/home/surajit/Downloads/20210507/eovsa_data/time_'+time_slice+'_scaled.fits']

model=Model(model_file)
spectrum=Spectrum(spectrum_files,xmin,ymin,xmax,ymax,lowest_freq,highest_freq)
spectrum.read_map()


#spec=spectrum.spectrum
#plt.imshow(spec[0,25,:,:],origin='lower')
#plt.show()

start=time.time()
gs_minimizer=GS_minimizer(model,spectrum,sys_error=sys_error,rms_thresh=rms_thresh,min_freq_num=min_freq_num)
fitted=gs_minimizer.fit_data()
end=time.time()
print (end-start)


spectrum_shape=np.shape(spectrum.spectrum)

print (fitted.shape,spectrum_shape)

param_maps=np.zeros((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3],model.num_params))
chi_map=np.zeros((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3]))

for t in range(spectrum_shape[0]):
	for y1 in range(spectrum_shape[2]):
		for x1 in range(spectrum_shape[3]):
			for m in range(model.num_params):
				param_maps[t,y1,x1,m]=model.param_vals[m][int(fitted[t,y1,x1,m])]
			chi_map[t,y1,x1]=fitted[t,y1,x1,model.num_params]

param_names=model.param_names

hf=h5py.File("fitted_param_maps_"+time_slice+"_python.hdf5",'w')
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

