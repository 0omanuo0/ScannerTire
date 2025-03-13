from src.Scanner import Scanner, Parameters
from compute_dxData import processFrame, compute_dxData_cython # type: ignore

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from picamera2 import Picamera2
from libcamera import controls 

import timeit


picam = Picamera2()
config = picam.create_video_configuration(
        main={"size": (2304, 1296), "format": "YUV420"})
# set camera configuration iso, shutter speed...
picam.configure(config)

# success = picam.autofocus_cycle()
# if(not success):
#     print("Autofocus failed")
#     exit(1) 
picam.set_controls({"FrameRate": 56.0,"AnalogueGain": 4.0, "ExposureTime": 100000, "AfMode": controls.AfModeEnum.Continuous})
picam.start()

parameters = Parameters(calibration_images="calibration_images/*.jpg", theta=30, L=10, resolution=(1296, 2304))
sc = Scanner(ignoreCamera=True)
frames = []

iso = 4.0
shutter_speed = 100000

# set slider opencv
cv2.namedWindow('Data', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Data', 1000, 500)
cv2.createTrackbar('ISO', 'Data', 0, 15, lambda x: None)
cv2.createTrackbar('Shutter Speed', 'Data', 1000, 100000, lambda x: None)
# cv2.createTrackbar('ColorFilter1', 'Data', 0, 255, lambda x: None)
# cv2.createTrackbar('ColorFilter2', 'Data', 0, 255, lambda x: None)
cv2.setTrackbarPos('ISO', 'Data', 1)
cv2.setTrackbarPos('Shutter Speed', 'Data', 10000)

tan_30 = np.tan(np.pi / 6)

time1 = cv2.getTickCount()
t1 = timeit.default_timer()

while True:

    iso = cv2.getTrackbarPos('ISO', 'Data')
    shutter_speed = cv2.getTrackbarPos('Shutter Speed', 'Data')
    # color_filter1 = cv2.getTrackbarPos('ColorFilter1', 'Data')
    # color_filter2 = cv2.getTrackbarPos('ColorFilter2', 'Data')

    picam.set_controls({"AnalogueGain": iso, "ExposureTime": shutter_speed})

    f = picam.capture_array()
    if f is None:
        continue

    f = cv2.cvtColor(f, cv2.COLOR_YUV2BGR_I420)
    
    f = np.rot90(f, 1)
    t2 = timeit.default_timer()
    data = sc.processFrame(f)
    t3 = timeit.default_timer()

    # show the data in to of f in green
    overlay = f.copy()
    overlay[data > 0] = (0, 255, 0)
    
    # overlay the fps, ms and ticks
    time2 = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (time2 - time1)
    
    ms = (time2 - time1) * 1000.0 / cv2.getTickFrequency()
    cv2.putText(overlay, f"{fps:.2f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"{ms:.4f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Processing time: {(t3 - t2)*1000:.4f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Total time: {timeit.default_timer() - t1:.2f} s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    time1 = time2
    cv2.imshow('Data', overlay)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        # rotate data
        data = np.rot90(data, 1)
        # plt.plot(dxData)
        # plt.show()
        dxData_points = sc.getPoints(data)
        transformed_dxData = cv2.perspectiveTransform(dxData_points, sc.H_total)
        dxP = transformed_dxData[:, :, 1]
        dzP = dxP / tan_30
        
        plt.plot(dzP)
        # # use mouse pointer to see the value of dzP
        mplcursors.cursor(hover=True)
        
        plt.show()
        frames.append(dzP)
        print(len(frames))
    if key == ord('p'):
        if len(frames) > 0:
            xshape = len(frames[0])
            yshape = len(frames)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(0, xshape)
            y = np.arange(0, yshape)
            x, y = np.meshgrid(x, y)
            z = np.array(frames)
            z2 = z.copy()
            z2 = cv2.GaussianBlur(z2, (11, 11), sigmaX=0, sigmaY=0)
            z2 = z2.reshape(yshape, xshape)
            # ax.plot_surface(x, y, z, cmap='plasma')
            ax.plot_surface(x, y, z2, cmap='viridis')
            mplcursors.cursor(hover=True)
            plt.show()
            
            

cv2.destroyAllWindows()
