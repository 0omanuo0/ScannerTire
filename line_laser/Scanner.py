import cv2
import numpy as np
import timeit
import time
from enum import Enum

import serial

class Parameters:
    def __init__(self, fov: float, theta: float, L: float, resolution: tuple[int, int] = (1280, 720)):
        
        self.fov = fov
        self.theta = theta
        self.L = L
        self.resolution = resolution
        
        self.x = resolution[0]
        self.y = resolution[1] // 2
        
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
    
    
    DEFAULT_PARAMETERS = Parameters(fov=60, theta=0, L=1000, resolution=(480, 640))
        
    class Filters(Enum):
        RAW = 0
        DEPTH_VALUES = 1
        DISTANCE_VALUES = 2

    def __init__(self, parameters: Parameters = DEFAULT_PARAMETERS, serial_port:str = None):
        
        # initialize serial port
        if serial_port != None:
            if not isinstance(serial_port, str):
                raise TypeError("serial_port must be a string")
            self.ser = serial.Serial('COM7', 9600)
            data = self.__waitResponseSerial()
            if data is not None:
                if not data.startswith("START STEPPER ON LOOP"):
                    raise ValueError("Invalid serial port")
            print("Serial port connected")
            self.ser.write(b"l2")
            time.sleep(1)
            
        # load params
        self.params = parameters
        
        # load camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Camera connected")
        # set the capture resolution to 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.params.resolution[0])
        print("Scanner initialized")
        
        

    def __waitResponseSerial(self, timeout:int = 2):
        t1 = timeit.default_timer()
        timeout_loop = False
        while (self.ser.in_waiting == 0):
            if timeit.default_timer() - t1 > timeout:
                timeout_loop = True
                break
            pass
        if timeout_loop:
            print("Timeout")
            return None
        return self.ser.readline().decode('utf-8')

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
                
                # Calculate the depth value (D) for this column
                D = (self.params.L * np.sin(self.params.theta_rad)) / np.sin(phi - self.params.theta_rad) if phi - self.params.theta_rad != 0 else np.inf
                depth_values.append(D)

            else:
                depth_values.append(np.nan)  # No laser spot found in this column
        
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
    
    def sendSteps(self, steps: int):
        """
        Send the number of steps to the Arduino.
        
        Parameters
        ----------
        steps : int
            The number of steps to send to the Arduino.
        """
        if not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        
        
        steps = f"step:{steps}\n"
        self.ser.write( str.encode(steps, 'ascii') )
        
    def getSteps(self, filter:Filters = 0)->list[np.ndarray]:
        """
        Get the number of steps from the Arduino.
        
        Parameters
        ----------
        handler : callable, optional
            The handler to process the frames from the Arduino.
        
        returns
        -------
        list[np.ndarray]
            The list of images from the Arduino.
        """
        
        # if handler != None:
        #     # if handler not return a np.ndarray 
        #     if not isinstance(handler(np.zeros((4, 4))), np.ndarray):
        #         raise TypeError("handler must return a numpy array")
        
        
        
        # read the first line (number of steps)
        steps = self.ser.readline().decode('ascii').strip()
        
        frames = []
        for i in range(int(steps)):
            data = self.__waitResponseSerial()
            
            if(data == "step:1\r\n"):
                frame = self.getFrame()
                if filter == Scanner.Filters.RAW:
                    frames.append(frame)
                elif filter == Scanner.Filters.DEPTH_VALUES:
                    frames.append(self.getDepthValues(frame))
                elif filter == Scanner.Filters.DISTANCE_VALUES:
                    frames.append(self.calculate_depth(self.processFrame(frame)))
        return frames

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'ser'):
            self.ser.close()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scanner = Scanner(serial_port="COM7")
    scanner.sendSteps(100)
    frames = scanner.getSteps(Scanner.Filters.DEPTH_VALUES)
    
    frames = np.array(frames)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(frames.shape[1])
    y = np.arange(frames.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, frames, cmap='viridis')
    plt.show()
    