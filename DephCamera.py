import pyrealsense2 as rs
import numpy as np
import cv2


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

    def toNumpy(self):
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

    
