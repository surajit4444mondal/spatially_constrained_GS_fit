Extracting Smooth Parameter Images using Spectral Imaging

Here I present a new software to do gyrosynchrotron fitting and obtain smooth parameter images. The algorithm also takes into the fact that the angular resolution is different for different frequencies. Hence in the last step of the algorithm, we actually generate model images using the trial parameter maps and smooth them with the appropriate resolution. At this point, we only support a circular gaussian kernel. However, in future, a asymmetric kernel would be supported. Then we compare the predicted images with the observed images.

The code has 11 essential inputs

xmin,ymin,xmax,ymax: The X and Y coordinates in the image in pixel units which would be used for fitting. 
		      (xmin,ymin) and (xmax,ymax) corresponds to bottom left and top right of the fitting region.

lowest_freq, highest_freq: Global minimum and maximum frequency in GHz which would be used for fitting. Please note
			    that ExtraSPICI would search within this range to find the appropriate frequency range
			    so that the spectrum can be modelled using a homogenous source model.

spectrum_files: Give a list of files. Each file is a composite of images at all frequencies. The frequencies are  
		 assumed to be sorted in ascending order. While it is envisioned that ultimately we want to make a 
		 movie of parameters, at this point, only one timeslice is supported downstream.
		 
smooth_lengths: Give a list of smooth lengths. This is the box size (smooth_length x smooth_length). It is advised
		 not to use smooth length much larger than the maximum angular resolution in absence of a valid 
		 reason. In the example provided, I use [0.5 x resolution, resolution]

sys_error: Fractional systematic uncertainty in flux. The actual error is estimated as rms^2+ sys_error^2 x spectrum value ^2

rms_thresh: SNR threshold to be considered a detection

min_freq_num: Minimum number of frequencies needed to do a fit.

discontinuity_thresh: The threshold in terms of box standard deviation, a parameter value can be different from the local median.
			If larger, I detect it as a discontinuity.
			
max_dist_parameter_space: Maximum distance between two members of a cluster in the parameter space. It is natural that if we
			   increase this, the number of clusters would decrease and the number of members will increase. This might
			   lead to a less smooth map.
			   
outfile: The file where the parameter maps and other information would be written.

resolution: FWHM of the gaussian kernel (resolution element) in pixel units. This should be array of size equal to the number of
	     frequencies.
	     
model: Please use a structure same as that provided in the example. Let us assume that there are $N$ parameters to be fitted. 
	Each parameter $N_i$ can take $M_i$ values. Additionally let us assume that the number of observed frequencies is $N_f$. 
	Then the model cube has the shape equal to $(M_0, M_1, M_2, ... , M_N, N_f)$. Other variables of the Model object should
	also be same as that provided in the example.
	
smoothness_enforcer: This parameter determines what weight should be given to the gradient. 

The code which the user needs to modify is make_param_maps_python_cython_combination.py .

Before running the python code first run,

python3 setup.py build_ext --inplace

Next run

python3 make_param_maps_python_cython_combination.py


