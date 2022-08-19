import h5py
import numpy as np

class GS_minimizer:
	def __init__(self,model_structure,spectral_structure,sys_error=0.1,rms_thresh=3,min_freq_num=10):
		self.model_structure=model_structure
		self.spectral_cube=spectral_structure
		self.sys_error=sys_error
		self.rms_thresh=rms_thresh
		self.min_freq_num=min_freq_num
		
	def fit_data(self):
		shape=self.spectral_cube.spectrum.shape
		cube=self.spectral_cube.spectrum
		err_cube=self.spectral_cube.error
		model=self.model_structure.model
		param_vals=self.model_structure.param_vals
		param_names=self.model_structure.param_names
		num_params=self.model_structure.num_params
		freqs=self.model_structure.freqs
		
		fitted=np.zeros((shape[1],shape[2],shape[3],num_params+1))
		print ("Starting minimisation")
		for t in range(shape[0]):
			for y1 in range(shape[2]):
				print (y1)
				for x1 in range(shape[3]):
					pos_freq=np.where((freqs>self.spectral_cube.lower_freq[t,y1,x1])&(freqs<self.spectral_cube.upper_freq[t,y1,x1]))[0]
					low_ind=pos_freq[0]
					high_ind=pos_freq[-1]
					spectrum=cube[t,low_ind:high_ind+1,y1,x1]
					rms=err_cube[t,low_ind:high_ind+1]
					sys_err=self.sys_error*spectrum
					error=np.sqrt(rms**2+sys_err**2)
					pos=np.where(spectrum>self.rms_thresh*rms)[0]
					if len(pos)<self.min_freq_num:
					    fitted[y1,x1,:]=0.0000
					    continue
						
					residual=np.sum((model[...,low_ind:high_ind+1]-spectrum)**2/error**2,axis=num_params)  
					
					### assumes that the first axis of model is the frequency axis
					min_residual=np.min(residual)
					pos=np.where(abs(residual-min_residual)<1e-5)
					print (pos)
					if len(pos[0])>1:
					    print ("Multiple vals in "+str(y1)+" "+str(x1))
					for i in range(num_params):
						fitted[y1,x1,i]=param_vals[i][pos[i][0]]
					fitted[t,y1,x1,num_params]=min_residual
					del pos_freq,low_ind,high_ind
					break
				break
		return fitted,param_names		
