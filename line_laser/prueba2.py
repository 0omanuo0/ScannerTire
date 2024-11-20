import numpy as np
import cv2
from src.Calibration import calibrateCamera
from src.Scanner import Scanner
import timeit
from compute_dxData import compute_dxData_cython # type: ignore
# Parámetros conocidos
# import numba
#config.DEBUG = True
H_total = calibrateCamera()
tan_30 = np.tan(np.pi/6)


# Assuming 'data' is already defined
allDxData = []
window_size = 5  # Adjust the window size as needed

sc = Scanner(cameraIndex=0, ignoreCamera=True)
# create 100 random frames of rgb data 1920x1080
data = np.random.randint(0, 255, (100, 1080, 1920, 3), dtype=np.uint8)
# iterate over the frames
for d in data:
    f = np.rot90(d, 2)
    t1 = timeit.default_timer()
    data = sc.processFrame(f)
    t2 = timeit.default_timer()
    print("time: ", t2-t1)



exit()
# data = cv2.merge([data, data, data])
# data = np.concatenate((data, f), axis=1)
# cv2.imshow('Data', data)
# cv2.waitKey(0)
data = np.load('dataframes2.npy')
for d in data:
    t1 = timeit.default_timer()
    d = np.transpose(d)
    # print(d.shape)
    assert isinstance(d, np.ndarray), "d debe ser un np.ndarray"
    assert d.dtype in [np.uint8, np.int8, np.int32], "wrong dtype"
    dxData = compute_dxData_cython(d)
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
    # print(dzP.shape)