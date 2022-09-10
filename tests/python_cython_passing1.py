import numpy as np
import return_structure_from_cython as rst

num=10
x=np.zeros((num,num))
y=np.zeros(num)
	
rst.test_function(x,y,num)

print (x)
print (y)
