from DephCamera import DephCamera, Frame
import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

plt.ion()

def applyFilter(image:np.ndarray, percentile:int)->np.ndarray:
    threshold = np.percentile(image, percentile)
    image[image > threshold] = 0
    return image

@jit(nopython=True)
def getSections(image: np.ndarray, sections: int) -> np.ndarray:
    mean_depth_image = np.zeros(image.shape, dtype=np.float64)
    
    mean_depth_graph = np.zeros(sections)
    
    max_depth = np.max(image)
    height = image.shape[0]

    for i in range(sections):
        start_col = i * (image.shape[1] // sections)
        end_col = (i + 1) * (image.shape[1] // sections)
        region = image[:, start_col:end_col]

        # Calcular la media solo para valores mayores que 0 en la región actual
        total = 0
        count = 0
        for row in range(region.shape[0]):
            for col in range(region.shape[1]):
                if region[row, col] > 0:
                    total += region[row, col]
                    count += 1

        mean_depth = total / count if count > 0 else 0
        mean_depth_graph[i] = mean_depth

        bar_height = int((mean_depth_graph[i] / max_depth if max_depth > 0 else 0) * image.shape[0])
        
        # Crear grafico de barras
        for row in range(region.shape[0]):
            for col in range(region.shape[1]):
                if ( height - row ) < bar_height:
                    mean_depth_image[row, start_col:end_col] = mean_depth_graph[i]
                else:
                    mean_depth_image[row, start_col:end_col] = 0
    
                
    return mean_depth_image


def main():
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # add a trackbar to change the threshold value
    cv2.createTrackbar('contrast', 'RealSense', 100, 1000, lambda x: x)
    cv2.createTrackbar('percentile', 'RealSense', 50, 100, lambda x: x)
    
    # add switch to enable or disable the temporal filter
    # cv2.createTrackbar('Temporal Filter', 'RealSense', 0, 1, lambda x: x)
    
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
                mean_depth_colormap2 = mean_depth_colormap.processFrame(getSections, (200,))
                
                
                
                mean_depth_colormap = mean_depth_colormap.convertTocv2(scale=contrast)
                mean_depth_colormap2 = mean_depth_colormap2.convertTocv2(scale=contrast)
                                
                images = mean_depth_colormap.Stack(mean_depth_colormap2)
                # images = depth_colormap.Stack(mean_depth_colormap)
    
                # Show images
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
            else:
                print("No depth image")
                break



if __name__ == "__main__":
    main()