from DephCamera import DephCamera, Frame, Graph
import cv2
import numpy as np
from numba import jit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from other_filters import applyFilter, getSections, getSectionsCenter

def generateFromSections(sections : list[np.ndarray], n_sections:int)->np.ndarray:
    x_coords = list(np.linspace(1, n_sections*10, n_sections))
    linfit = interp1d(x_coords, np.vstack(sections), axis=0)
    return linfit(list(np.linspace(1,n_sections*10, n_sections*10))) 
    

def main():
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    
    cv2.createTrackbar('contrast', 'RealSense', 100, 1000, lambda x: x)
    cv2.createTrackbar('height', 'RealSense', 200, 480, lambda x: x)
    
    graph = Graph()
    
    sections : list[np.ndarray] = []

    with DephCamera() as cam:
        cam.initStream()
        while True:
            # check for keyboard input q or close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cv2.getWindowProperty('RealSense', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            depth_image = cam.getDepthImage()
            if depth_image is not None:
                
                height = cv2.getTrackbarPos('height', 'RealSense')
                contrast = cv2.getTrackbarPos('contrast', 'RealSense')/1000
                
                mean_depth_colormap = Frame.asFrame(depth_image.toNumpy())
                
                mean_depth_colormap = mean_depth_colormap.processFrame(applyFilter, (80,))
                if mean_depth_colormap is None:
                    continue
                
                mean_depth_colormap2, graph_data = getSectionsCenter(mean_depth_colormap.toNumpy(), height)
                # mean_depth_colormap2, graph_data = getSections(mean_depth_colormap.toNumpy(), 200)
                
                mean_depth_colormap2 = Frame(mean_depth_colormap2)
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    # show the graph with matplotlib and show the valleys and peaks
                    graph.showGraph(graph_data)
                elif cv2.waitKey(1) & 0xFF == ord('v'):
                    print("Saving section")
                    sections.append(graph_data)
                elif cv2.waitKey(1) & 0xFF == ord('o'):
                    n_sections = len(sections)
                    if n_sections < 5:
                        print("Not enough sections")
                        continue
                    
                    final = generateFromSections(sections, n_sections)
                    graph.saveGraph(final, "final.stl", overwrite=True)
                    graph.showImg(final)
                elif cv2.waitKey(1) & 0xFF == ord('p'):
                    n_sections = len(sections)
                    if n_sections < 5:
                        print("Not enough sections")
                        continue
                    
                    final = generateFromSections(sections, n_sections)
                    graph.saveGraph(final, "final.stl", overwrite=True)
                    graph.show3dGraph(final)
                    
                
                mean_depth_colormap = mean_depth_colormap.convertTocv2(scale=contrast)
                mean_depth_colormap2 = mean_depth_colormap2.convertTocv2(scale=contrast)
                                
                images = mean_depth_colormap.Stack(mean_depth_colormap2)
    
                # Show images
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
            else:
                print("No depth image")
                break




if __name__ == "__main__":
    main()