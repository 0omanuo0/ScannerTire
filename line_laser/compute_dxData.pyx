# compute_dxData.pyx

import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython cimport boundscheck, wraparound, cdivision

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