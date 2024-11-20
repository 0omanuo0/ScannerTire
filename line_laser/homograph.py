import cv2
import numpy as np
import glob

def calibrateCamera():
    # Known parameters
    square_size = 25  # Size of a square in mm
    pattern_size = (10, 7)  # Internal corners in the chessboard

    # Generate 3D real-world points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []  # 3D and 2D points

    # Load calibration images
    calibration_images = glob.glob('calibration_images/*.jpg')
    if not calibration_images:
        print("No calibration images found.")
        return None

    for fname in calibration_images:
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # Refine corner accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            img_shape = gray.shape[::-1]
        else:
            print(f"Chessboard corners not detected in {fname}")

    if not objpoints:
        print("No valid chessboard corners found in any image.")
        return None

    # Camera calibration
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Use the first image for homography calculation
    corners_undistorted = cv2.undistortPoints(imgpoints[0], mtx, dist, P=mtx)
    H, _ = cv2.findHomography(corners_undistorted, objpoints[0][:, :2])

    # Calculate the translation matrix to set the origin to the chessboard center
    center_idx = len(objpoints[0]) // 2
    center_world = objpoints[0][center_idx, :2]
    T = np.array([[1, 0, -center_world[0]], [0, 1, -center_world[1]], [0, 0, 1]])

    # Combine translation and homography
    H_final = np.dot(T, H)

    return H_final

H_total = calibrateCamera()


from Scanner import Scanner, Parameters
import matplotlib.pyplot as plt

p = Parameters(theta=30, fov=60, L=300, resolution=(1920, 1080))
scanner = Scanner(cameraIndex=2, parameters=p)
frames = []
dataframes = []

# get image from camera (realtime)
cap = cv2.VideoCapture(2)
# set the resolution
cap.set(3, 1920)

while True:
    f = scanner.getFrame()
    # rotate 180 degrees
    f = np.rot90(f, 2)
    # display the frame and the data stacked
    d = scanner.processFrame(f)
    cv2.imshow('Data', np.concatenate((f, cv2.merge([d, d, d])), axis=1))
    
    frames.append(f)
    dataframes.append(d)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # elif cv2.waitKey(1) & 0xFF == ord('s'):
    #     # Matriz total de homografía
    #     H_total = H_translate.dot(H)

    #     # Crear un array vacío para almacenar los valores de dx
    #     dx_array = np.zeros(d.shape[1], dtype=np.float32)

    #     # Iterar sobre cada columna de la imagen 'd'
    #     for col in range(d.shape[1]):
    #         # Encontrar el primer píxel no nulo en la columna
    #         non_zero_rows = np.where(d[:, col] > 0)[0]
            
    #         if len(non_zero_rows) > 0:
    #             row = non_zero_rows[0]  # Coordenada de fila del primer píxel no nulo
    #             point = np.array([[[col, row]]], dtype=np.float32)  # Punto en coordenadas de imagen
                
    #             # Calcular la transformación de perspectiva para este punto
    #             transformed_point = cv2.perspectiveTransform(point, H_total)
                
    #             # Calcular dx como la diferencia entre el punto transformado y el original
    #             dx = transformed_point[0, 0, 0] - point[0, 0, 0]
    #         else:
    #             # Si no hay píxel no nulo, dx = 0
    #             dx = 0
            
    #         # Asignar dx al array
    #         dx_array[col] = dx

    #     # Imprimir o guardar los resultados
        
    #     # convert to inverse geometric triangulation of the laser 
    #     # FORMULA: dz = dx/tan(theta)
    #     dzPoints = []
    #     for point in dx_array:
    #         dz = point / np.tan(np.radians(30))
    #         dzPoints.append(dz)
    #     plt.plot(dx_array)
    #     plt.plot(dzPoints)
    #     plt.show()
    #     # frames.append(dzPoints)
    #     # print(len(frames))
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # save the frames into data.npy
        np.save('frames.npy', frames)
        np.save('dataframes.npy', dataframes)

        # if len(frames) > 0:
        #     fd = np.array(frames)
        #     plt.plot(fd[0])
        #     # print(fd[0])
        #     print(fd.shape)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     x = np.arange(fd.shape[1])
        #     y = np.arange(fd.shape[0])
        #     X, Y = np.meshgrid(x, y)
        #     ax.plot_surface(X, Y, fd, cmap='viridis')
        #     # set the z axis limits
        #     ax.set_zlim(0, -400)
        #     plt.show()

   

    # # display in a hystogram the depth values with the frame (opencv)
    # black = np.zeros((f.shape[0], f.shape[1], 3), np.uint8)
    # for i, dz in enumerate(dzPoints):
    #     cv2.line(black, (i, 0), (i, int(dz)), (255, 255, 255), 1)
    
    # # join data and f, first convert data to 3 channels
    # black = np.concatenate((black, f), axis=1)
    # cv2.imshow('Data', black)
    







