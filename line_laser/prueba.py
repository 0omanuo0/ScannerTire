from Scanner import Scanner, Parameters
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a new scanner and show live the data with opencv
p = Parameters(theta=30, fov=60, L=300, resolution=(480, 640))
scanner = Scanner(cameraIndex=0, parameters=p)
frames = []

while True:
    # Get the data from the scanner
    f = scanner.getFrame()
    # rotate 180 degrees
    f = np.rot90(f, 2)
    # Show the data with opencv
    data = scanner.processFrame(f)
    # join data and f, first convert data to 3 channels
    data = cv2.merge([data, data, data])
    data = np.concatenate((data, f), axis=1)
    cv2.imshow('Data', data)
    
    # Wait for a key press
    key = cv2.waitKey(1)
    # If the key is 'q' break the loop
    if key == ord('q'):
        break
    if key == ord('s'):
        frames.append(scanner.getDepthValues(f))
        print(len(frames))
    if key == ord('p'):
        if len(frames) > 0:
            fd = np.array(frames)
            plt.plot(fd[0])
            # print(fd[0])
            print(fd.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(fd.shape[1])
            y = np.arange(fd.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, fd, cmap='viridis')
            # set the z axis limits
            ax.set_zlim(0, -400)
            plt.show()

# Destroy all the windows
cv2.destroyAllWindows()