import numpy as np
from inputs import alpha


def calc_gradient(param):
'''
This function retruns the modulus of the gradient of
the param map at each location.
Assume that param is a 2D array, with each axis being
a spatial dimension. 
If array is 3D, assume that the first axis=0 is the time axis.
The other 2 are spatial axes
'''
	grad=np.gradient(param)
	if param.ndim==2:
		space_grad=np.sqrt(grad[0]**2+grad[1]**2)
		time_grad=None
		return space_grad,time_grad
	elif param.ndim==3:
		space_grad=np.sqrt(grad[1]**2+grad[2]**2)
		time_grad=np.abs(grad[0])
		return space_grad,time_grad
	elif param.ndim==1:
		space_grad=np.abs(grad[0])
		time_grad=None
		return space_grad,time_grad
	elif param.ndim==4:
		space_grad=np.sqrt(grad[1]**2+grad[2]**2+grad[3]**2)
		time_grad=np.abs(grad[0])
		return space_grad,time_grad
	else:
		raise RuntimeError("Array dimension not supported")
		



		
