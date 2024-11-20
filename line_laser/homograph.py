import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from Calibration import calibrateCamera
from InterativePlot import interactive_plot

# Parámetros conocidos


H_total = calibrateCamera()



# load numpy array from .npy file
data = np.load('dataframes.npy')
# Assuming 'data' is already defined
allDxData = []
window_size = 5  # Adjust the window size as needed


for d in data:
    d = np.transpose(d)

    dxData = []
    for i, j in enumerate(d):
        k = 0
        nonzero_indices = np.nonzero(j)[0]
        if nonzero_indices.size != 0 and nonzero_indices[0] < 450:
            k = int(nonzero_indices[0])

        dxData.append([i, k])

    dxData_points = np.array(dxData, dtype=np.float32).reshape(-1, 1, 2)

    # Perform the perspective transformation
    transformed_dxData = cv2.perspectiveTransform(dxData_points, H_total)
    dxP = transformed_dxData[:, :, 1]

    # calculate dz = dx / tan(30)
    dzP = dxP / np.tan(np.pi/6)
    allDxData.append(dzP)



# interactive_plot(allDxData[0], title='Interactive Plot')

# copy allDxData[0] to dz
import random
dz = allDxData[random.randint(0, len(allDxData) - 1)]
# resize (1920, 1) to 1920
dz = dz.reshape(1920)

import numpy as np

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


# convert shape 1920 to 500-1250
xshape = len(range(500, 1250))

result = []
for dz in allDxData:
    dz = dz.reshape(1920)
    dz_interpolated = removeSpikes(dz, th=10, min_length=20)
    dz_interpolated = removeSpikes(dz_interpolated, th=6, min_length=20)
    # crop from 500 to 1250
    dz_interpolated = dz_interpolated[500:1250]
    
    result.append(dz_interpolated)



# plot in 3d the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, xshape)
y = np.arange(0, 273)
x, y = np.meshgrid(x, y)
z = np.array(result)
z2 = z.copy()
z2 = cv2.GaussianBlur(z2, (11, 11), sigmaX=0, sigmaY=0)
# z = z.reshape(273, xshape)
z2 = z2.reshape(273, xshape)
# ax.plot_surface(x, y, z, cmap='plasma')
ax.plot_surface(x, y, z2, cmap='viridis')
plt.show()
