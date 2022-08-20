import h5py
import numpy as np
import matplotlib.pyplot as plt


cython_file='fitted_param_maps.hdf5'
python_file='fitted_param_maps_python.hdf5'

key='chi_sq'

cython_hf=h5py.File(cython_file)
cython_param=np.array(cython_hf[key])[0,:,:]
cython_hf.close()

python_hf=h5py.File(python_file)
python_param=np.array(python_hf[key])[0,:,:]
python_hf.close()

fig=plt.figure()
ax=fig.add_subplot(131)
plt.imshow(cython_param,origin='lower')
plt.colorbar()
plt.title("Cython output")

ax=fig.add_subplot(132)
plt.imshow(python_param,origin='lower')
plt.colorbar()
plt.title("Python output")

ax=fig.add_subplot(133)
plt.imshow(python_param/cython_param,origin='lower')
plt.colorbar()
plt.title("Python/Cython")

plt.savefig("chisq_comparison.png")
plt.show()
