import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
import os
from stl import mesh

class Graph:
    def __generateMeshGrid(self, data:np.ndarray, remove_black_lines : bool = False)->mesh.Mesh:
        def remove_black_lines(image):
            # Find non-zero rows and columns
            rows = np.where(np.any(image > 300, axis=1))[0]
            cols = np.where(np.any(image > 300, axis=0))[0]
            
            # Crop image
            cropped_image = image[rows.min():rows.max()+1, cols.min():cols.max()+1]
            
            return cropped_image

        image = remove_black_lines(data)
        # Get image dimensions
        height, width = image.shape
        plt.imshow(image)
        plt.show()
        
        # Generate grid of vertices
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Scale intensity values to define vertex heights
        Z = image 
        
        # Create vertices array
        vertices = np.zeros((height * width, 3))
        vertices[:, 0] = np.ravel(X)
        vertices[:, 1] = np.ravel(Y)
        vertices[:, 2] = np.ravel(Z)
        
        # Create mesh
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                # Define vertices for each quad
                v0 = i * width + j
                v1 = (i + 1) * width + j
                v2 = (i + 1) * width + j + 1
                v3 = i * width + j + 1
                
                # Add two triangles to form the quad
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
        
        # Create mesh object
        mesh_obj = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                mesh_obj.vectors[i][j] = vertices[f[j], :]
        
        return mesh_obj
             
        
        
    def __init__(self) -> None:
        self.thread : threading.Thread = threading.Thread()
        self.isShowing : bool = False
        
    def saveGraph(self, data : np.ndarray, filename:str, overwrite:bool=False)->None:
        extension = filename.split(".")[-1]
        
        if os.path.exists(filename) and not overwrite:
            counter = 1
            while os.path.exists(f"{filename} ({counter}).{extension}"):
                counter += 1
            filename = f"{filename} ({counter}).{extension}"        
            
        if extension == "stl":
            stl_mesh = self.__generateMeshGrid(data).save(filename)
        elif extension == "npy":
            np.save(filename, data)
            # read the file to check if it was saved correctly
            a = np.load(filename)
            print(a.shape)
        
            
    def show3dGraph(self, data:np.ndarray)->None:
        if self.isShowing:
            return
        self.thread = threading.Thread(target=self.__show3dGraph, args=(data,))
        self.thread.start()
        
    def __show3dGraph(self, data:np.ndarray)->None:
        self.isShowing = True
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, data.shape[1], data.shape[1])
        y = np.linspace(0, data.shape[0], data.shape[0])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, data)
        plt.show()
        # wait for the user to close the graph
        plt.pause(0.01)
        plt.close()
        
        self.isShowing = False
        
    def __isShowing(self)->bool:
        if self.thread in threading.enumerate():
            self.isShowing = True
        else:
            self.isShowing = False
        return self.isShowing
        
    def showImg(self, img:np.ndarray)->None:
        if self.isShowing:
            return
        self.thread
        self.thread = threading.Thread(target=self.__showImg, args=(img,))
        self.thread.start()
    
    def __showImg(self, img:np.ndarray)->None:
        plt.imshow(img)
        plt.show()
        # wait for the user to close the graph
        plt.pause(0.01)
        plt.close()
        self.isShowing = False
        
    def showGraph(self, data:np.ndarray)->None:
        if self.isShowing:
            return
        self.thread = threading.Thread(target=self.__showGraph, args=(data,))
        self.thread.start()
        
    def __showGraph(self, graph_data:np.ndarray):
        self.isShowing = True
        peaks = argrelmax(graph_data, order=10)
        valleys = argrelmin(graph_data, order=10)
        
        #remove the first and last peak and valley
        # peaks = (peaks[0][1:-1],)
        # valleys = (valleys[0][1:-1],)
        
        plt.plot(graph_data)
        # show also the value peaks and valleys
        plt.scatter(peaks, graph_data[peaks], color='red')
        plt.scatter(valleys, graph_data[valleys], color='green')
        
        for i in range(len(peaks[0])):
            x = peaks[0][i]
            y = graph_data[peaks[0][i]]
            value = f"{graph_data[peaks[0][i]]:.2f}"
            plt.text(x, y, value, fontsize=9, color='red')
        for i in range(len(valleys[0])):
            x = valleys[0][i]
            y = graph_data[valleys[0][i]]
            value = f"{graph_data[valleys[0][i]]:.2f}"
            plt.text(x, y, value, fontsize=9, color='green')
        
        plt.show()
        # wait for the user to close the graph
        plt.pause(0.01)
        # pause the thread
        plt.waitforbuttonpress()
        plt.close()
        self.isShowing = False


class Frame(np.ndarray):
    def asFrame(a:any)->"Frame":
        return Frame(np.asarray(a))
    
    def __init__(self, input_array, info=None):
        self.info = info
        self._input_array = input_array
        self._info = info
        
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj
    
    def convertTocv2(self, scale=1):
        return Frame(cv2.convertScaleAbs(self._input_array, alpha=scale))
    
    def convertTocv2ColorMap(self, colorMap:int=1, scale=1):
        return Frame(cv2.applyColorMap(self.convertTocv2(scale), colorMap))
    
    def processFrame(self, processing_frame:callable, args:tuple )->"Frame":
        return Frame(processing_frame(self._input_array, *args))
        
    def Stack(self, other):
       return np.hstack((self._input_array, other))

    def toNumpy(self)->np.ndarray:
        return self._input_array
    

class DephCamera:
    def __init__(self)->None:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
    
    def initStream(self, resolution=(640, 480), format=rs.format.z16, fps=30)->None:
        
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], format, fps)
        self.pipeline.start(self.config)
    
    def getProfile(self)->rs.video_stream_profile:
        return self.pipeline.get_active_profile().get_stream(rs.stream.depth)
    
    def getDepthImage(self)->Frame:
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            # Convert images to numpy arrays
            depth_image = Frame.asFrame(depth_frame.get_data())
            return depth_image
        except Exception as e:
            print(e)
            return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.stop()
        return False

    
