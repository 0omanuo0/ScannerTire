# compute_dxData.pyx

import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython cimport boundscheck, wraparound, cdivision
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
import cv2


# Desactiva comprobaciones para mejorar el rendimiento
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[np.int32_t, ndim=2] compute_dxData_cython(np.ndarray[np.uint8_t, ndim=2] d):
    cdef int rows = d.shape[0]
    cdef int cols = d.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] dxData = np.empty((rows, 2), dtype=np.int32)
    
    # Usar memory views tipados para acceso rápido
    cdef np.uint8_t[:, :] d_view = d
    cdef int[:, :] dx_view = dxData
    
    cdef int i, j, k
    
    # Liberar el GIL para permitir la ejecución paralela
    with nogil:
        for i in prange(rows):
            k = 0
            for j in range(cols):
                if d_view[i, j] != 0:
                    k = j
                    break
            dx_view[i, 0] = i
            dx_view[i, 1] = k

    return dxData

# def processFrame(self, frame: np.ndarray) -> np.ndarray: frame is hsv image 


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[np.uint8_t, ndim=2] processFrame(np.ndarray[np.uint8_t, ndim=3] frame):
    cdef int height = frame.shape[0]
    cdef int width = frame.shape[1]
    
    # Convertir la imagen a HSV
    cdef np.ndarray[np.uint8_t, ndim=3] hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir el rango de color rojo en HSV
    cdef np.ndarray[np.uint8_t, ndim=1] lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    # Crear máscaras para detectar el color ro jo
    cdef np.ndarray[np.uint8_t, ndim=2] mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    cdef np.ndarray[np.uint8_t, ndim=2] mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cdef np.ndarray[np.uint8_t, ndim=2] mask = cv2.bitwise_or(mask1, mask2)
    
    # Obtener el canal rojo directamente
    cdef np.ndarray[np.uint8_t, ndim=2] red_channel = frame[:, :, 2]
    
    # Aplicar la máscara al canal rojo
    cdef np.ndarray[np.uint8_t, ndim=2] masked_red = cv2.bitwise_and(red_channel, red_channel, mask=mask)
    
    # Encontrar el valor máximo y el índice por columna
    cdef np.ndarray[np.uint8_t, ndim=1] max_values = masked_red.max(axis=0)
    cdef np.ndarray[np.intp_t, ndim=1] max_indices = masked_red.argmax(axis=0)
    
    # Identificar las columnas donde el valor máximo es mayor que cero
    cdef np.ndarray[np.intp_t, ndim=1] nonzero_cols = np.nonzero(max_values)[0]
    
    # Crear la imagen de resultado
    cdef np.ndarray[np.uint8_t, ndim=2] result_luminous = np.zeros((height, width), dtype=np.uint8)
    
    # Marcar los puntos más luminosos en el resultado
    result_luminous[max_indices[nonzero_cols], nonzero_cols] = 255
    
    return result_luminous




