import sys
sys.path.append('/home/surajit/pygsfit/pygsfit/')
from utils import gstools,ndfits
from inputs import sys_error,lowest_freq,highest_freq,rms_thres

class Spectrum:
	def __init__(self,files,resolution=None,lower_freq=None,\
			upper_freq=None,upper_limit=None,smooth_length=4):
	'''
	filename is a list of files to use, sorted in increasing order of time
	'''
		self.files=files
		self.resolution=resolution
		self.lower_freq=lower_freq
		self.upper_freq=upper_freq
		self.smooth_length=smooth_length
		self.spectrum=None
		self.error=None
	
	def read_spectrum(self):
		files=self.files
		num_files=len(files)
		for i in range(num_files):
			filename=files[i]
			meta,rdata=ndfits.read(filename)
			freqs=meta['ref_cfreqs']*1e-9  ### convert to GHz
			pgdata = rdata[0, :, :, :]
			shape=pgdata.shape
			if i==0:
				spectrum=np.zeros((num_files,shape[0],shape[1],shape[2]))
				error=np.zeros((num_files,shape[0]))
				lower_freq=np.zeros((num_files,shape[0]))
				upper_freq=np.zeros((num_files,shape[1]))
				upper_limit=np.zeros((num_files,shape[0],shape[1],shape[2]),dtype=bool)
				
			spectrum[i,:,:,:]=pgdata
			rms=np.nanstd(rdata[0,:,0:shape[1]/3,0:shape[2]],axis=(1,2))  ### taking rms along
								### a horizontal stripe
			error[i,:]=rms
		self.spectrum=spectrum
		self.error=error
	
	
	
	
	
			
			
			
		
			
