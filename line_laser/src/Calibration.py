import cv2
import numpy as np
import glob

def calibrateCamera(
        square_size: int = 25,
        pattern_size: tuple = (10, 7),
        calibration_images_str: str = 'calibration_images/*.jpg'
):

    # Generate 3D real-world points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []  # 3D and 2D points

    # Load calibration images
    calibration_images = glob.glob(calibration_images_str)
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