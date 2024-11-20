import cv2
import numpy as np
import timeit
import time
from enum import Enum

import serial

class Parameters:
    def __init__(self, fov: float, theta: float, L: float, resolution: tuple[int, int] = (1280, 720)):
        
        self.fov = fov
        # theta is the angle of the camera with respect to the laser
        self.theta = theta
        # L is the distance between the camera and the laser
        self.L = L
        self.resolution = resolution
        
        
        self.x = resolution[0]
        self.y = resolution[1] // 2 # center of the image
        
        self.fov_rad = np.deg2rad(fov)
        self.theta_rad = np.deg2rad(theta)
        self.angle_per_pixel = self.fov_rad / resolution[1]




class Scanner:
    """
    A class to represent a line laser scanner.
    
    Attributes
    ----------
    DEFAULT_PARAMETERS : Parameters
        The default parameters for the scanner.
    Filters : Enum
        The filters for the scanner.
        
    Methods
    -------
    calculate_depth(image:np.ndarray) -> np.ndarray:
        Calculate the depth value for each column with a white pixel (laser spot).
    processFrame(frame:np.ndarray) -> np.ndarray:
        Process the frame to detect the laser spot.
    getFrame() -> np.ndarray:
        Get a frame from the camera.
    getProcessedFrame() -> np.ndarray:
        Get a processed frame with the laser spot.
    getDepthValues(frame:np.ndarray) -> np.ndarray:
        Get the depth values for the laser spot in the frame.
    sendSteps(steps:int):
        Send the number of steps to the Arduino.
    getSteps(handler:callable) -> np.ndarray:
        Get the number of steps from the Arduino.
    """
    
    
    DEFAULT_PARAMETERS = Parameters(fov=60, theta=30, L=1000, resolution=(480, 640))
        
    class Filters(Enum):
        RAW = 0
        DEPTH_VALUES = 1
        DISTANCE_VALUES = 2

    def __init__(self, parameters: Parameters = DEFAULT_PARAMETERS, serial_port:str = None, cameraIndex:int = 2):
                   
        # load params
        self.params = parameters
        
        # load camera
        self.cap = cv2.VideoCapture(cameraIndex, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Desactivar exposición automática
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        print("Camera connected")
        # set the capture resolution to params.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.params.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params.resolution[1])
        print("Scanner initialized")
        
        

    def calculate_depth(self, image:np.ndarray) -> np.ndarray:
        """
        Calculate the depth value for each column with a white pixel (laser spot).
        
        Parameters
        ----------
        image : np.ndarray
            The image with the laser spot.
        
            
        Returns
        -------
        list
            The depth value for each column with a white pixel (laser spot).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if image.ndim != 2:
            raise ValueError("image must be a 2D array")
        
        if image.shape[0] != self.params.resolution[0] or image.shape[1] != self.params.resolution[1]:
            raise ValueError("image must have the same resolution as the camera")
        
        # Find the row index of the white pixel (value 255) in each column
        depth_values:list[float] = []
        for col in range(self.params.x):
            # Find the index of the white pixel in the column
            row_indices = np.where(image[:, col] == 255)[0]
            if len(row_indices) > 0:
                # Assume the first white pixel is the laser spot
                row_index = row_indices[0]
                
                # Calculate the observed angle (phi) for this column
                pixel_offset = row_index - (self.params.y) 
                phi = pixel_offset * self.params.angle_per_pixel
                
                # print(f"Pixel offset: {pixel_offset}, Phi: {phi}")

                # Calculate the depth value (D) for this column
                # the formula is D = (L * sin(theta)) / sin(phi - theta)
                D = (self.params.L * np.sin(self.params.theta_rad)) / np.sin(phi - self.params.theta_rad) if phi - self.params.theta_rad != 0 else np.inf
                
                depth_values.append(D)

            else:
                depth_values.append(np.nan)  # No laser spot found in this column'
        
        return np.array(depth_values)
    
    def processFrame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame to detect the laser spot.
        
        Parameters
        ----------
        frame : np.ndarray
            The frame to process.
            
        Returns
        -------
        np.ndarray
            The processed frame with the laser spot.
        """
        
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define el rango de color rojo en HSV
        # Nota: El color rojo en HSV puede estar en dos rangos debido a su naturaleza circular
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Crea una máscara para los píxeles que están dentro de los rangos definidos
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Aplica la máscara a la imagen original para obtener solo los píxeles rojos
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # filtro solo rojo a 
        result_red = result.copy()
        result_red[:, :, 0] = 0
        result_red[:, :, 1] = 0
            
        gray = cv2.cvtColor(result_red, cv2.COLOR_BGR2GRAY)

        # Crear una imagen negra para el resultado
        result_luminous = np.zeros_like(gray)

        # Buscar el punto más luminoso de cada columna
        for col in range(gray.shape[1]):
            column = gray[:, col]
            max_idx = np.argmax(column)
            if column[max_idx] > 0:  # Solo considerar si hay un punto rojo
                result_luminous[max_idx, col] = 255
        
        return result_luminous
    
    def getFrame(self)->np.ndarray:
        """
        Get a frame from the camera.
        
        Returns
        -------
        np.ndarray
            The frame from the camera.
        """
        _, frame = self.cap.read()
        return frame
    
    def getProcessedFrame(self) -> np.ndarray:
        """
        Get a processed frame with the laser spot.
        
        Returns
        -------
        np.ndarray
            The processed frame with the laser spot.
        """
        frame = self.getFrame()
        return self.processFrame(frame)
    
    def getDepthValues(self, frame) -> np.ndarray:
        """
        Get the depth values for the laser spot in the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            The frame with the laser spot.
        
        Returns
        -------
        list
            The depth values for the laser spot in the frame.
        """
        processed_frame = self.processFrame(frame)
        return self.calculate_depth(processed_frame)
    

    def processFrames(self, frames: list[np.ndarray], filter:Filters = 0)->list[np.ndarray]:
        """
        Process the frames from the Arduino.
        
        Parameters
        ----------
        frames : list[np.ndarray]
            The list of frames from the Arduino.
        filter : Filters
            The filter to apply to the frames.
        
        returns
        -------
        list[np.ndarray]
            The list of processed frames.
        """
        
        processed_frames = []
        for frame in frames:
            if filter == Scanner.Filters.RAW:
                processed_frames.append(frame)
            elif filter == Scanner.Filters.DEPTH_VALUES:
                processed_frames.append(self.getDepthValues(frame))
            elif filter == Scanner.Filters.DISTANCE_VALUES:
                processed_frames.append(self.calculate_depth(self.processFrame(frame)))
        return processed_frames

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'ser'):
            self.ser.close()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # scanner = Scanner(serial_port="COM7")
    # scanner.sendSteps(100)
    # frames = scanner.getSteps(Scanner.Filters.DEPTH_VALUES)
    
    # frames = np.array(frames)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(frames.shape[1])
    # y = np.arange(frames.shape[0])
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, frames, cmap='viridis')
    # plt.show()


    ## instead of using the serial port to comunicate take frames with the terminal
    scanner = Scanner()
    # just to try take a frame and process it
    frame = scanner.getFrame()
    processed_frame = scanner.processFrame(frame)
    depth_values = scanner.calculate_depth(processed_frame)
    # print(depth_values)
    plt.imshow(processed_frame)
    plt.show()
    plt.imshow(frame)
    plt.show()

    