# Taken from https://www.experoinc.com/post/a-quick-note-on-gpu-accuracy-and-double-precision

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from matplotlib import pyplot as plt

N = 10000000

d = np.random.rand(N)
f = d.astype(np.float32)
g = gpuarray.to_gpu(f)

pd = np.dot(d,d)
pf = np.dot(f,f)
pg = gpuarray.dot(g,g).get()

ferr = np.abs(pd-pf)
gerr = np.abs(pd-pg)

print("True sum:",pd)
print("CPU 32bit precision:",pf,"error:",ferr)
print("GPU 32bit precision:",pg,"error:",gerr)

plt.bar(['cpu error','gpu error'], [ferr, gerr], color='bg')
plt.show()
