import h5py
import numpy as np

class GS_minimizer:
	def __init__(self,model_structure,spectral_structure,sys_error=0.1,rms_thresh=3,min_freq_num=10):
		self.model_structure=model_structure
		self.spectral_cube=spectral_structure
		self.sys_error=sys_error
		self.rms_thresh=rms_thresh
		self.min_freq_num=min_freq_num
		num_params=self.model_structure.num_params
		freqs=self.model_structure.freqs
		self.num_freqs=len(freqs)
		self.num_params=num_params
		pos_freq=np.where((freqs>self.spectral_cube.lowest_freq)&(freqs<self.spectral_cube.highest_freq))[0]
		shape=self.spectral_cube.spectrum.shape
		low_ind=np.ones((shape[0],shape[2],shape[3]))*pos_freq[0]
		high_ind=np.ones((shape[0],shape[2],shape[3]))*pos_freq[-1]
		self.lower_freq_ind=low_ind
		self.higher_freq_ind=high_ind
		
	def check_homgenous_source(self,t,y1,x1):
		smooth_length=3
		smoothed_spectrum=np.convolve(self.spectrum,np.ones(smooth_length),mode='same')
		smoothed_spectrum[0:max(smooth_length//2,self.lower_freq_ind[t,y1,x1])]=0.0
		smoothed_spectrum[min(self.higher_freq_ind[t,y1,x1],num_freqs-smooth_length//2):num_freqs]=0.0
		max_loc=np.argsort(self.spectrum)
		
		num_freqs=self.num_freqs
		
		j=max_loc-smooth_interval
	
		while j>smooth_interval:
			if smoothed[j]<1e-7:
				break
			if (smoothed_spectrum[j-1]-smoothed_spectrum[j])/smoothed_spectrum[j]>0.05:
				self.lower_freq_ind[t,y1,x1]=j
				break
			j=j-1
			
		j=max_loc+1+smooth_interval
		upper_freq_ind[0]=j
		
		while j<num_freqs-2-smooth_interval:
			if smoothed_spectrum[j]<1e-7:
				upper_freq_ind[0]=j
			if (smoothed_spectrum[j+1]-smoothed_spectrum[j])/smoothed_spectrum[j]>0.05:
				self.higher_freq_ind[t,y1,x1]=j
				break
			j=j+1
		return
		
	def fit_data(self):
		shape=self.spectral_cube.spectrum.shape
		cube=self.spectral_cube.spectrum
		err_cube=self.spectral_cube.error
		model=self.model_structure.model
		
		fitted=np.zeros((shape[0],shape[2],shape[3],self.num_params+1))
		print ("Starting minimisation")
		first_try=True
		for t in range(shape[0]):
			for y1 in range(shape[2]):
				print (y1)
				for x1 in range(shape[3]):
					#if first_try==True:
					#	check_homogenous_source(self.t,y1,x1)
					low_ind=2#self.lower_freq_ind[t,y1,x1]
					high_ind=38#self.higher_freq_ind[t,y1,x1]
					spectrum=cube[t,low_ind:high_ind+1,y1,x1]
					rms=err_cube[t,low_ind:high_ind+1]
					sys_err=self.sys_error*spectrum
					error=np.sqrt(rms**2+sys_err**2)
					pos=np.where(spectrum>self.rms_thresh*rms)[0]
					if len(pos)<self.min_freq_num:
						fitted[t,y1,x1,:]=0.0000
						continue
					
					mid_ind=(low_ind+high_ind)//2
					ratio=model[...,mid_ind]/spectrum[mid_ind]
					pos=np.where((ratio<2) & (ratio>0.5))
					if len(pos[0])==0:
						fitted[t,y1,x1,:]=0.0000
						continue
					#residual=np.sum((model[...,low_ind:high_ind+1]-spectrum)**2/error**2,axis=num_params)  
					### assumes that the first axis of model is the frequency axis
					residual=np.sum((model[pos][:,low_ind:high_ind+1]-spectrum)**2/error**2,axis=1)
					
					min_residual=np.min(residual)
					pos1=np.where(abs(residual-min_residual)<1e-5)
					#if len(pos1[0])>1:
					 #  print ("Multiple vals in "+str(y1)+" "+str(x1))
					for i in range(self.num_params):
						fitted[t,y1,x1,i]=pos[i][pos1[0]]
					fitted[t,y1,x1,self.num_params]=min_residual
					del low_ind,high_ind
		return fitted	
