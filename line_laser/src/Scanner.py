import cv2
import numpy as np
import timeit
import time
from enum import Enum
from src.Calibration import calibrateCamera


def removeSpikes(dz, th=6, min_length=10):
    dz = np.array(dz)
    n = len(dz)
    if n == 0:
        return dz.tolist()

    # Inicializar variables
    tramos_start = []
    tramos_end = []
    tramo_sizes = []
    tramo_start = 0

    # Identificar los límites de los tramos y su tamaño
    for i in range(1, n):
        if abs(dz[i] - dz[i - 1]) >= th:
            tramos_start.append(tramo_start)
            tramos_end.append(i - 1)
            tramo_sizes.append(tramos_end[-1] - tramos_start[-1] + 1)
            tramo_start = i
    # Agregar el último tramo
    tramos_start.append(tramo_start)
    tramos_end.append(n - 1)
    tramo_sizes.append(tramos_end[-1] - tramos_start[-1] + 1)

    # Inicializar variables para interpolación
    result = dz.copy()
    idx = 0
    last_large_segment_end = None
    small_segments_indices = []

    while idx < len(tramos_start):
        start = tramos_start[idx]
        end = tramos_end[idx]
        tramo_len = tramo_sizes[idx]

        if tramo_len < min_length:
            # Acumular índices de tramos pequeños consecutivos
            small_segments_indices.append((start, end))
        else:
            # Si hay tramos pequeños acumulados, interpolar a través de ellos
            if small_segments_indices:
                # Determinar los límites de interpolación
                if last_large_segment_end is not None:
                    left_value = dz[last_large_segment_end]
                else:
                    left_value = dz[small_segments_indices[0][0]]

                right_value = dz[start]

                # Obtener el rango completo para interpolar
                interp_start = small_segments_indices[0][0]
                interp_end = small_segments_indices[-1][1]
                total_len = interp_end - interp_start + 1

                # Interpolar a través de todos los tramos pequeños acumulados
                interpolated_values = np.linspace(left_value, right_value, total_len)
                result[interp_start:interp_end+1] = interpolated_values

                # Limpiar acumulador de tramos pequeños
                small_segments_indices = []

            # Actualizar el fin del último tramo grande
            last_large_segment_end = end

        idx += 1

    # Si los últimos tramos son pequeños, interpolar hasta el final
    if small_segments_indices:
        if last_large_segment_end is not None:
            left_value = dz[last_large_segment_end]
        else:
            left_value = dz[small_segments_indices[0][0]]

        right_value = dz[small_segments_indices[-1][1]]  # Usar el último valor disponible
        interp_start = small_segments_indices[0][0]
        interp_end = small_segments_indices[-1][1]
        total_len = interp_end - interp_start + 1

        # Interpolar o extrapolar
        interpolated_values = np.linspace(left_value, right_value, total_len)
        result[interp_start:interp_end+1] = interpolated_values

    return result.tolist()

class Parameters:
    """
    A class to represent the parameters of the scanner.

    Attributes
    ----------
    calibration_images : str
        The path to the calibration images.
    theta : float
        The angle of the camera with respect to the laser.
    L : float
        The distance between the camera and the laser.
    pattern : tuple[int, int]
        The chessboard pattern of the calibration images.
    pattern_size : float
        The size of the squares in the pattern.
    resolution : tuple[int, int]
    
    """
    def __init__(self, calibration_images: str, theta: float, L: float, pattern : tuple[int, int] = (10, 7), pattern_size:float=25.0, resolution: tuple[int, int] = (1280, 720)):
        
        # theta is the angle of the camera with respect to the laser
        self.theta = theta
        # L is the distance between the camera and the laser
        self.L = L
        self.resolution = resolution
        
        
        self.x = resolution[0]
        self.y = resolution[1] // 2 # center of the image
        
        self.calibration_images = calibration_images
        self.pattern = pattern
        self.pattern_size = pattern_size

        self.theta_rad = np.deg2rad(theta)




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
    
    
    DEFAULT_PARAMETERS = Parameters(calibration_images="calibration_images/*.jpg", theta=30, L=10, resolution=(1920, 1080))
        
    class Filters(Enum):
        RAW = 0
        DEPTH_VALUES = 1
        DISTANCE_VALUES = 2

    def __init__(self, parameters: Parameters = DEFAULT_PARAMETERS, cameraIndex:int = 2):
                   
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
        
        # load calibration
        self.H_total = calibrateCamera(
                                square_size=self.params.pattern_size, 
                                pattern_size=self.params.pattern, 
                                calibration_images_str=self.params.calibration_images
                            )
        print("Calibration loaded")

    
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
    
    def calculate_depth(self, image:np.ndarray, trimm:bool=False, remove_spikes:bool=True) -> np.ndarray:
        d = np.transpose(image)

        dxData = []
        for i, j in enumerate(d):
            k = 0
            nonzero_indices = np.nonzero(j)[0]
            if nonzero_indices.size != 0 and nonzero_indices[0] < 450: 
                k = int(nonzero_indices[0])

            dxData.append([i, k])

        dxData_points = np.array(dxData, dtype=np.float32).reshape(-1, 1, 2)

        # Perform the perspective transformation
        transformed_dxData = cv2.perspectiveTransform(dxData_points, self.H_total)
        dxP = transformed_dxData[:, :, 1]

        # calculate dz = dx / tan(30)
        dzP = dxP / np.tan(self.params.theta_rad)

        xshape = 1920
        if trimm:
            dzP = dzP[500:1250]
            xshape = 750
        
        dz = dzP.reshape(xshape)
        if not remove_spikes:
            return dz
        dz_interpolated = removeSpikes(dz, th=10, min_length=20)
        dz_interpolated = removeSpikes(dz_interpolated, th=6, min_length=20)

        return dz_interpolated



    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        

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

    