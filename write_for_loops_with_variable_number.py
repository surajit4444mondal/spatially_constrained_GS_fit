import numpy as np

def for_loop(num_params,a):
	pos=np.where(a>0)[0]
	if len(pos)==num_params:
		print (a)
		return
	
	for i in range(1,4):
		if len(pos)==0:
			a[0]=i
		else:
			a[pos[-1]+1]=i
		for_loop(num_params,a)
	return
	
num_params=5
b=np.zeros(num_params)
for_loop(num_params,b)	
