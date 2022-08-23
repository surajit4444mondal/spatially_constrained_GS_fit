import sys
sys.path.append('/home/surajit/pygsfit/pygsfit/')
from utils import gstools,ndfits
import numpy as np


class Spectrum:
	def __init__(self,files,xmin,ymin,xmax,ymax,lowest_freq,\
			highest_freq,resolution=None,smooth_length=4):
		self.files=files
		self.resolution=resolution
		self.lower_freq=None
		self.upper_freq=None
		self.smooth_length=None
		self.spectrum=None
		self.error=None
		self.ymin=ymin
		self.ymax=ymax
		self.xmin=xmin
		self.xmax=xmax
		self.lowest_freq=lowest_freq
		self.highest_freq=highest_freq
	
	def read_map(self):
		files=self.files
		num_files=len(files)
		for i in range(num_files):
			filename=files[i]
			meta,rdata=ndfits.read(filename)
			freqs=meta['ref_cfreqs']*1e-9  ### convert to GHz
			pgdata = rdata[0, :, self.ymin:self.ymax+1, self.xmin:self.xmax+1]
			shape=pgdata.shape
			full_shape=rdata[0,...].shape
			if i==0:
				spectrum=np.zeros((num_files,shape[0],shape[1],shape[2]))
				error=np.zeros((num_files,shape[0]))
				
			spectrum[i,:,:,:]=pgdata
			rms=np.nanstd(rdata[0,:,0:full_shape[1]//3,0:full_shape[2]],axis=(1,2))  ### taking rms along
								### a horizontal stripe
			error[i,:]=rms
		
		self.spectrum=spectrum
		self.error=error
		
	
	
	
	
	
			
			
			
		
			
