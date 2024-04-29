from DephCamera import DephCamera, Frame
import cv2
import numpy as np
from numba import jit



def applyFilter(image:np.ndarray, percentile:int)->np.ndarray:
    threshold = np.percentile(image, percentile)
    image[image > threshold] = 0
    return image


def main():
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # add a trackbar to change the threshold value
    cv2.createTrackbar('contrast', 'RealSense', 100, 1000, lambda x: x)
    cv2.createTrackbar('percentile', 'RealSense', 50, 100, lambda x: x)
    
    # add switch to enable or disable the temporal filter
    cv2.createTrackbar('Temporal Filter', 'RealSense', 0, 1, lambda x: x)
    
    # temp = []

    with DephCamera() as cam:
        cam.initStream()
        while True:
            # check for keyboard input q or close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            depth_image = cam.getDepthImage()
            if depth_image is not None:
                
                # append the depth image to the temporal list and remove the first element if the list has more than 10 elements
                # temp.append(depth_image.toNumpy())
                # if len(temp) > 10:
                #     temp.pop(0)
                
                percentile = cv2.getTrackbarPos('percentile', 'RealSense')
                contrast = cv2.getTrackbarPos('contrast', 'RealSense')/1000
                # buttonState = cv2.getTrackbarPos('Temporal Filter', 'RealSense')
                
                depth_colormap = depth_image.convertTocv2(scale=contrast)
                
                mean_depth_colormap = Frame.asFrame(depth_image.toNumpy())
                # if buttonState:
                #     mean_depth_colormap = depth_colormap.processFrame(filterTemporal, (temp,))
                #     mean_depth_colormap = mean_depth_colormap.processFrame(removeLines, ())
                mean_depth_colormap = mean_depth_colormap.processFrame(applyFilter, (percentile,))
                
                mean_depth_colormap = mean_depth_colormap.convertTocv2(scale=contrast)
                                
                
                images = depth_colormap.Stack(mean_depth_colormap)
    
                # Show images
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
            else:
                print("No depth image")
                break



if __name__ == "__main__":
    main()