import numpy as np
from spectrum import Spectrum
import python_functions as pf
import timeit
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
max_dist_parameter_space=4
outfile="param_maps.hdf5"


cell=1.5  ###arcsec
freqs=np.loadtxt("/home/surajit/Downloads/20210507/eovsa_data/ctrl_freqs.txt")
num_freqs=np.size(freqs)
resolution=np.zeros(num_freqs)
for i in range(num_freqs):
	resolution[i]=(max(60/(freqs[i]*1e-3),3))/cell 


spectrum_files=['/home/surajit/Downloads/20210507/eovsa_data/time_'+time_slice+'_scaled.fits']

model=Model(model_file)
spectrum=Spectrum(spectrum_files,xmin,ymin,xmax,ymax,lowest_freq,highest_freq)
spectrum.read_map()


pf.main_func(xmin,ymin,xmax,ymax,lowest_freq,highest_freq,min_freq_num,spectrum,model,resolution, smooth_lengths,\
		sys_error=sys_error,rms_thresh=rms_thresh,discontinuity_thresh=discontinuity_thresh,\
		max_dist_parameter_space=max_dist_parameter_space, outfile=outfile)






