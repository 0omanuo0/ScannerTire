#load external webcam
import cv2
import numpy as np
import time
import timeit
from numba import jit
import matplotlib.pyplot as plt

# import serial lib
import serial

# set serial port com5 with baudrate 9600
ser = serial.Serial('COM7', 9600)


# load camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set the capture resolution to 1280x720
cap.set(3, 1280)

# define slider for the window to adjust the threshold



def processFrame(frame):
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



def calculate_depth(image:np.ndarray, fov:float, theta:float, L:float) -> list:
    """
    Calculate the depth value for each column with a white pixel (laser spot).
    
    Parameters
    ----------
    image : np.ndarray
        The image with the laser spot.
    fov : float
        The field of view of the camera (degrees).
    theta : float
        The angle of the camera (degrees).
    L : float
        The distance between the camera and the laser pointer (mm).
        
    Returns
    -------
    list
        The depth value for each column with a white pixel (laser spot).
    """
    # Convert angles from degrees to radians
    fov_rad : float = np.deg2rad(fov)
    theta_rad : float = np.deg2rad(theta)
    
    # Get the image dimensions
    rows, cols = image.shape
    
    # Calculate angle per pixel
    angle_per_pixel = fov_rad / cols
    
    # Find the row index of the white pixel (value 255) in each column
    depth_values:list[float] = []
    for col in range(cols):
        # Find the index of the white pixel in the column
        row_indices = np.where(image[:, col] == 255)[0]
        if len(row_indices) > 0:
            # Assume the first white pixel is the laser spot
            row_index = row_indices[0]
            
            # Calculate the observed angle (phi) for this column
            pixel_offset = row_index - (rows // 2) 
            phi = pixel_offset * angle_per_pixel
            
            # Calculate the depth value (D) for this column
            D = (L * np.sin(theta_rad)) / np.sin(phi - theta_rad) if phi - theta_rad != 0 else np.inf
            depth_values.append(D)

        else:
            depth_values.append(np.nan)  # No laser spot found in this column
    
    return depth_values



frames = []
# send serial "l2"
time.sleep(2)
ser.write(b"l2")


# display camera
while True:
    
    
    _, frame = cap.read()
    # Convierte la imagen al espacio de color HSV
    
    res = processFrame(frame)
    depths = calculate_depth(res, 60, 15, 15)
    depths = np.array(depths)
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        # send serial "step:300"
        ser.write(b"step:300")
        # wait for the response "step:1"
        for i in range(300):
            t1 = timeit.default_timer()
            timeout_loop = False
            while (ser.in_waiting == 0):
                if timeit.default_timer() - t1 > 2:
                    timeout_loop = True
                    break
                pass
            if timeout_loop:
                print("Timeout")
                break
            data = ser.readline().decode('utf-8')
            
            if(data == "step:1\r\n"):
                print(data)
                _, frame = cap.read()
                # Convierte la imagen al espacio de color HSV
                
                res = processFrame(frame)
                depths = calculate_depth(res, 60, 15, 15)
                depths = np.array(depths)
                
                frames.append(depths)
        
        # plot the array of frames as 3d
        frames = np.array(frames)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(frames.shape[1])
        y = np.arange(frames.shape[0])
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, frames, cmap='viridis')
        plt.show()
        
        frames = []
            
    if cv2.waitKey(1) & 0xFF == ord('s'):
        max_depth = np.nanmax(depths)
        min_depth = np.nanmin(depths)
        
        # interactive plot to get the point value with the mouse
        plt.plot(depths, picker=6)
        plt.xlabel('Column')
        plt.ylabel('Depth (mm)')
        plt.title('Depth values for each column')
        plt.grid()
        
        
        
        def clickCursor(event):
            artist = event.artist
            ind = event.ind[0]
            x = artist.get_xdata()[ind]
            
            # get the closest point to the cursor from the values
            y = depths[round(x)]
            
            print(f'Column: {round(x)} Depth: {y} mm')
        
        cid = plt.gcf().canvas.mpl_connect('button_press_event', clickCursor)
        
        plt.show()
        
                
        
        
    frame[res > 0] = [0, 255, 0]
    frame = np.hstack((frame, cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)))
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# command to connect serial cmd: python3.11.exe -m serial.tools.miniterm COM6 9600