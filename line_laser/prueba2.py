import numpy as np
import cv2
from src.Calibration import calibrateCamera
import timeit
from compute_dxData import compute_dxData_numba
# Parámetros conocidos
# import numba
#config.DEBUG = True
H_total = calibrateCamera()
tan_30 = np.tan(np.pi/6)


# load numpy array from .npy file
data = np.load('dataframes2.npy')
print(data.shape)
print(data.dtype)

# Assuming 'data' is already defined
allDxData = []
window_size = 5  # Adjust the window size as needed

def compute_dxData(d):
    d = np.array(d)
    nonzero_indices = np.argmax(d != 0, axis=1)

    # Identificar filas donde hay al menos un valor no nulo
    valid_rows = (d != 0).any(axis=1)

    # Asignar 0 donde no se cumple la condición
    k_values = np.zeros_like(nonzero_indices)
    k_values[valid_rows] = nonzero_indices[valid_rows]

    # Generar el array final directamente como NumPy
    return np.column_stack((np.arange(d.shape[0]), k_values))





for d in data:
    t1 = timeit.default_timer()
    d = np.transpose(d)
    print(d.shape)
    assert isinstance(d, np.ndarray), "d debe ser un np.ndarray"
    assert d.dtype in [np.uint8, np.int8, np.int32], "wrong dtype"
    dxData = compute_dxData(d)
    t2 = timeit.default_timer()
    dxData_points = np.array(dxData, dtype=np.float32).reshape(-1, 1, 2)
    t3 = timeit.default_timer()
    # Perform the perspective transformation
    transformed_dxData = cv2.perspectiveTransform(dxData_points, H_total)
    dxP = transformed_dxData[:, :, 1]
    t4 =  timeit.default_timer()
    # calculate dz = dx / tan(30)
    dzP = dxP / tan_30
    allDxData.append(dzP)
    t5 = timeit.default_timer()
    print("time: ", t2-t1, ",", t3-t2, ",", t4-t3, ",", t5-t4)
    print(dzP.shape)