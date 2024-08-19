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

dsum = d.sum()
fsum = f.sum()
gsum = gpuarray.sum(g).get()

ferr = np.abs(dsum-fsum)
gerr = np.abs(dsum-gsum)

print("True sum:",dsum)
print("CPU 32bit precision:",fsum,"error:",ferr)
print("GPU 32bit precision:",gsum,"error:",gerr)

plt.bar(['cpu error','gpu error'], [ferr, gerr], color='bg')
plt.show()
