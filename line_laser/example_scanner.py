from src.Scanner import Scanner, Parameters
import cv2
import numpy as np
import matplotlib.pyplot as plt


sc = Scanner(cameraIndex=0)
frames = []

while True:

    f = sc.getFrame()
    f = np.rot90(f, 2)
    data = sc.processFrame(f)

    data = cv2.merge([data, data, data])
    data = np.concatenate((data, f), axis=1)
    cv2.imshow('Data', data)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        frames.append(sc.calculate_depth(f))
        print(len(frames))
    if key == ord('p'):
        if len(frames) > 0:
            xshape = len(frames[0])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(0, xshape)
            y = np.arange(0, 273)
            x, y = np.meshgrid(x, y)
            z = np.array(frames)
            z2 = z.copy()
            z2 = cv2.GaussianBlur(z2, (11, 11), sigmaX=0, sigmaY=0)
            # z = z.reshape(273, xshape)
            z2 = z2.reshape(273, xshape)
            # ax.plot_surface(x, y, z, cmap='plasma')
            ax.plot_surface(x, y, z2, cmap='viridis')
            plt.show()

cv2.destroyAllWindows()