import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generate the 3D real-world coordinates for the chessboard corners
# Equation: Object points represent (x, y, 0) grid of the chessboard
square_size = 10  # Square size in mm
pattern_size = (10, 7)  # Chessboard dimensions
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Load the calibration image and detect chessboard corners
image = cv2.imread('calibration_images/WIN_20241115_11_42_36_Pro.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
if not ret:
    raise ValueError("No corners detected in the calibration image.")

# Calibrate the camera to get the distortion parameters
objpoints = [objp]  # List of object points
imgpoints = [corners]  # List of image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Load a frame and apply lens distortion correction
data = np.load('frames.npy')
frame = data[0]  # Load the first frame
undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

# Define the function to map a point from the output to the input image
def convert_pt(point, width, height, f, r, y_c):
    """
    Convert a pixel from the output image to the corresponding position in the input image.
    Applies cylindrical equations to map coordinates.
    
    Equations:
    - Circle: x^2 + (z - z0)^2 = r^2
    - Quadratic for z: z = (-b + sqrt(b^2 - 4ac)) / 2a
      a = (ypc^2 / f^2) + 1
      b = -2 * z0
      c = z0^2 - r^2
    - Final mapping: 
      x_orig = x_pc * z / f
      y_orig = y_pc * z / f
    """
    pc = np.array([point[0] - width / 2, point[1] - y_c])
    omega = width / 2
    z0 = f - np.sqrt(r * r - omega * omega)
    a = pc[1]**2 / (f**2) + 1
    b = -2 * z0
    c = z0**2 - r**2
    zc = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # Solve quadratic for z
    final_point = np.array([pc[0] * zc / f, pc[1] * zc / f])
    final_point[0] += width / 2
    final_point[1] += y_c
    return final_point

# Apply cylindrical projection correction
def cylindrical_correction(image, f, r, y_c):
    """
    Apply cylindrical projection correction.
    
    Each pixel in the output image is mapped back to the input using:
    - Cylindrical projection equations for (x, y, z)
    - Bilinear interpolation for smooth remapping
    """
    height, width = image.shape[:2]
    corrected_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            current_pos = convert_pt((x, y), width, height, f, r, y_c)
            top_left = (int(current_pos[0]), int(current_pos[1]))
            if top_left[0] < 0 or top_left[0] >= width - 1 or top_left[1] < 0 or top_left[1] >= height - 1:
                continue
            dx = current_pos[0] - top_left[0]
            dy = current_pos[1] - top_left[1]
            weight_tl = (1.0 - dx) * (1.0 - dy)
            weight_tr = dx * (1.0 - dy)
            weight_bl = (1.0 - dx) * dy
            weight_br = dx * dy
            for c in range(image.shape[2]):
                corrected_image[y, x, c] = (
                    weight_tl * image[top_left[1], top_left[0], c]
                    + weight_tr * image[top_left[1], top_left[0] + 1, c]
                    + weight_bl * image[top_left[1] + 1, top_left[0], c]
                    + weight_br * image[top_left[1] + 1, top_left[0] + 1, c]
                )
    return corrected_image

# Calculate parameters for the cylindrical transformation
height, width = undistorted.shape[:2]
r = width / 2  # Cylinder radius in pixels (assumes physical width = 652 mm)
y_c = height - 280  # Center of curvature (280 px from bottom)
f = r + np.sqrt(r**2 - (width / 2)**2)  # Estimate focal length

# DRAW LINES
y1 = 0
y2 = undistorted.shape[0] - 1
x1 = undistorted.shape[1] // 2 - 300
x2 = undistorted.shape[1] // 2 + 300

# Dibujar las líneas
cv2.line(undistorted, (x1, y1), (x1, y2), (0, 255, 0), 3)
cv2.line(undistorted, (x2, y1), (x2, y2), (0, 255, 0), 3)

# Apply cylindrical correction and display the result
corrected_image = cylindrical_correction(undistorted, f, r, y_c)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.title('Lens-Corrected Image')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title('Cylindrically Corrected Image')
plt.show()
