import numpy as np
import matplotlib.pyplot as plt
import time
#from get_model import Model
from spectrum import Spectrum
from gs_minimizer import GS_minimizer
import calculate_chi_square as chi_mod
import h5py

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


xmin=550
ymin=520
xmax=551
ymax=521
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
'''
hf=h5py.File("fitted_param_maps_python.hdf5",'w')
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
'''
