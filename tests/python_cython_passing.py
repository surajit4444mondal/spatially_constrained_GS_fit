import numpy as np
import change_numpy_arrays_inplace_in_cython as rst

num=10
x=np.zeros(num)
y=np.zeros(num)
	
rst.test_function(x,y,num)

print (x)
print (y)
