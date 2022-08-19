import numpy as np
import matplotlib.pyplot as plt

from get_model import Model
from spectrum import Spectrum
from gs_minimizer import GS_minimizer
import calculate_chi_square as chi_mod

time_slice='190200_190210'
model_file='/home/surajit/Downloads/20210507/eovsa_data/model_gs_spectra.hdf5'
outfile='model_paramaters_'+time_slice+'.hdf5'
lower_freq=2
upper_freq=14


xmin=539
ymin=503
xmax=548
ymax=513
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

gs_minimizer=GS_minimizer(model,spectrum,sys_error=sys_error,rms_thresh=rms_thresh,min_freq_num=min_freq_num)
fitted=gs_minimizer.fit_data()


param_lengths=np.zeros(len(model.param_names),dtype=int)
shape=np.shape(model.model)

for i in range(len(model.param_names)):
	param_lengths[i]=shape[i]

spectrum_shape=np.shape(spectrum.spectrum)
model1=np.ravel(model.model)

f1=chi_mod.compute_min_chi_square(model1,spectrum.spectrum,spectrum.error,spectrum.lower_freq,\
		spectrum.upper_freq,param_lengths,model.freqs,sys_error,rms_thresh,min_freq_num,\
		len(model.param_names), spectrum_shape[0],spectrum_shape[1],spectrum_shape[2],\
		spectrum_shape[3])

f1=f1.reshape((spectrum_shape[0],spectrum_shape[2],spectrum_shape[3],len(model.param_names)+1))
print (f1[0,0,0,:])
