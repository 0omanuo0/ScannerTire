# create a cell

#%%python
import numpy as np
import zlib
import timeit
import blosc
from src.Calibration import calibrateCamera
from src.Scanner import Scanner
from compute_dxData import compute_dxData_cython  # type: ignore
data = np.load("../dataframes2.npy")
# %%
data[0]
# get size in bytes of data[0]
import sys
d = data[0].copy()
print(sys.getsizeof(data)/1024)
print(sys.getsizeof(d)/1024)
print(sys.getsizeof(np.packbits(d))/1024)
t1 = timeit.default_timer()
print(sys.getsizeof(blosc.compress(np.packbits(d)))/1024)
t2 = timeit.default_timer()
print("time: ", t2-t1)
dxData = compute_dxData_cython(d)
dxData_points = np.array(dxData, dtype=np.float32).reshape(-1, 1, 2)
print(sys.getsizeof(blosc.compress(dxData_points))/1024)

# %%
data2 = np.load("../frames.npy")
d2 = data2[0].copy()
print(sys.getsizeof(d2)/1024/1024, "MB")
# %%
