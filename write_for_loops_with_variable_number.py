import numpy as np

'''
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



x=[[1,2,3],[1,2,3],[1,2,3]]

len_x=len(x)

lists=[]
for i in range(len_x):
	list1=x[i]
	length=len(list1)
	len_list_prev=len(lists)
	if len_list_prev!=0:
		list2=[i for i in lists]
		for j in range(length-1):
			for k in list2:
				lists.append(k)
		m=0
		print (lists)
		for k in list1:
			for i in range(m,len_list_prev*(m+1)):
				print (i)
				lists[i].append(k)
				print (lists)
			m+=len_list_prev
	else:
		for k in list1:
			lists.append([k])
			
	if i>0:
		break
print (lists)
print (len(lists))
'''	

x=[[1,2,3],[1,2,3,4],[1,2,3]]

len_x=len(x)

lists=[]
for i in range(len_x):
	list1=x[i]
	length=len(list1)
	len_list_prev=len(lists)
	if len_list_prev!=0:
		list2=[i for i in lists]
		for j in range(length-1):
			for k in list2:
				lists.append([p for p in k])
		m=0
		for k in list1:
			for i in range(len_list_prev*m,len_list_prev*(m+1)):
				lists[i].append(k)
			m+=1
	else:
		for k in list1:
			lists.append([k])
			
print (lists)
print (len(lists))

