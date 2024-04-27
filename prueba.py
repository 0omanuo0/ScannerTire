## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import timeit

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

time = timeit.default_timer()

try:
    while True:
        # check for keyboard input q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        
        # get the mean deph of every column and create a vector with the values
        mean_depth = np.mean(depth_image, axis=0)
        
            
        # calculate the mean of the mean depth for 20 regions of the image
        sections = 80
        regions = np.array_split(mean_depth, sections)
        mean_regions = [np.mean(region) for region in regions]
        # print 1s
        if timeit.default_timer() - time > 3:
            # print in cm
            print([round(i/100, 2) for i in mean_regions])
            time = timeit.default_timer()
            print(depth_image.shape)
            
        #visualize the regions as a horizontal line create other image with the same size of the depth image 
        mean_depth_image = np.zeros(depth_image.shape)
        for i in range(sections):
            mean_depth_image[:, i*6:(i+1)*6] = mean_regions[i]
        

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)
        mean_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(mean_depth_image, alpha=0.2), cv2.COLORMAP_JET)
        
        images = np.hstack((depth_colormap, mean_depth_colormap))


        depth_colormap_dim = depth_colormap.shape

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()