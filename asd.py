if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scanner = Scanner()
    # create a scanner that given the input 'f' take a frame, and 'c' is for generate the scan
    key = input("Press 'f' to take a frame, 'c' to generate the scan or 'q' to quit: ")
    frames_deph = []
    while key != 'q':
        if key == 'f':
            frame = scanner.getFrame()
            # # rotate the frame 90 degrees
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            depth = scanner.getDepthValues(frame)
            frames_deph.append(depth)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #display a fraph of the depth values
            plt.figure()
            plt.plot(depth)
            plt.show()
        elif key == 'c':
            # DISPLAY IN 3D THE POINTS FRAMES_DEPH
            frames_deph_t = np.array(frames_deph)
            print(frames_deph_t.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(frames_deph_t.shape[1])
            y = np.arange(frames_deph_t.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, frames_deph_t, cmap='viridis')
            plt.show()
        elif key == 'k': # konsole python interpreter (input command to inspect variables)
            command = input("$: ")
            if command == 'q' or command == 'quit':
                break
            else:
                print(eval(command))


        key = input("Press 'f' to take a frame, 'c' to generate the scan or 'q' to quit: ")
    del scanner
