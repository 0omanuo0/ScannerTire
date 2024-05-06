from DephCamera import DephCamera, Frame
import cv2
import numpy as np
from numba import njit, jit



def applyFilter(image:np.ndarray, percentile:int)->np.ndarray:
    threshold = np.percentile(image, percentile)
    image[image > threshold] = 0
    return image

@jit(nopython=True)
def getSectionsCenter(image: np.ndarray, height_region: int, sections: int = 200) -> tuple[Frame, np.ndarray]:
    mean_depth_image = np.zeros(image.shape, dtype=np.float64)
    mean_depth_graph = np.zeros(sections)

    max_depth = np.max(image)
    height = image.shape[0]

    center = image.shape[0] // 2
    region_IMG = image[center - height_region // 2:center + height_region // 2]

    for i in range(sections):
        start_col = i * (region_IMG.shape[1] // sections)
        end_col = (i + 1) * (region_IMG.shape[1] // sections)
        region = region_IMG[:, start_col:end_col]

        total = 0
        count = 0
        for row in range(region.shape[0]):
            for col in range(region.shape[1]):
                if region[row, col] > 0:
                    total += region[row, col]
                    count += 1

        mean_depth = total / count if count > 0 else 0
        mean_depth_graph[i] = mean_depth

        bar_height = int((mean_depth / max_depth if max_depth > 0 and not np.isnan(mean_depth) else 0) * region_IMG.shape[0])

        for row in range(image.shape[0]):
            for col in range(start_col, end_col):  # Adjusted to update only the current section's region
                if ( height - row ) < bar_height:
                    mean_depth_image[row, start_col:end_col] = mean_depth_graph[i]
                else:
                    mean_depth_image[row, start_col:end_col] = 0

    return mean_depth_image, mean_depth_graph

    
    
    
@jit(nopython=True)
def getSections(image: np.ndarray, sections: int)->tuple[Frame, np.ndarray]:
    mean_depth_image = np.zeros(image.shape, dtype=np.float64)
    
    mean_depth_graph = np.zeros(sections)
    
    max_depth = np.max(image)
    height = image.shape[0]

    for i in range(sections):
        start_col = i * (image.shape[1] // sections)
        end_col = (i + 1) * (image.shape[1] // sections)
        region = image[:, start_col:end_col]

        # Calcular la media solo para valores mayores que 0 en la región actual
        total = 0
        count = 0
        for row in range(region.shape[0]):
            for col in range(region.shape[1]):
                if region[row, col] > 0:
                    total += region[row, col]
                    count += 1

        mean_depth = total / count if count > 0 else 0
        mean_depth_graph[i] = mean_depth

        bar_height = int((mean_depth_graph[i] / max_depth if max_depth > 0 else 0) * image.shape[0])
        
        # Crear grafico de barras
        for row in range(region.shape[0]):
            for col in range(region.shape[1]):
                if ( height - row ) < bar_height:
                    mean_depth_image[row, start_col:end_col] = mean_depth_graph[i]
                else:
                    mean_depth_image[row, start_col:end_col] = 0
    
                
    return mean_depth_image, mean_depth_graph












def filterTemporal(image:np.ndarray, temp:list)->np.ndarray:
    # calculate the mean of the temporal images
    mean_image = np.mean(temp, axis=0)
    return mean_image

@jit(nopython=True)
def conv2d(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    output = np.zeros_like(image)
    kh, kw = km // 2, kn // 2
    
    for i in range(kh, m - kh):
        for j in range(kw, n - kw):
            sum = 0
            for ii in range(km):
                for jj in range(kn):
                    sum += kernel[ii, jj] * image[i - kh + ii, j - kw + jj]
            output[i, j] = sum
    return output

@jit(nopython=True)
def removeLines(image):    
    kernel2 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])
    return conv2d(image, kernel2)







    
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])
    

@jit(nopython=True)
def apply_filter(image, kernel):
    # Obtiene las dimensiones de la imagen y el filtro
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Crea una matriz para almacenar la imagen filtrada
    filtered_image = np.zeros_like(image)
    
    # Aplica el filtro a la imagen
    for i in range(img_height):
        for j in range(img_width):
            # Calcula los límites de la región a convolucionar
            i_start = max(i - kernel_height // 2, 0)
            i_end = min(i + kernel_height // 2 + 1, img_height)
            j_start = max(j - kernel_width // 2, 0)
            j_end = min(j + kernel_width // 2 + 1, img_width)
            
            # Inicializa el valor del píxel filtrado
            filtered_pixel = 0.0
            
            # Realiza la convolución
            for m in range(i_start, i_end):
                for n in range(j_start, j_end):
                    # Calcula las coordenadas dentro del kernel
                    k_row = kernel_height // 2 - (i - m)
                    k_col = kernel_width // 2 - (j - n)
                    
                    # Suma el producto de la región de la imagen y el kernel
                    filtered_pixel += image[m, n] * kernel[k_row, k_col]
            
            # Asigna el valor del píxel filtrado a la imagen resultante
            filtered_image[i, j] = filtered_pixel
    
    return filtered_image


@njit
def calculate_mask(rows, cols, low_cutoff, high_cutoff):
    mask = np.zeros((rows, cols), np.uint16)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - rows / 2)**2 + (j - cols / 2)**2)
            if low_cutoff <= dist <= high_cutoff:
                mask[i, j] = 1
    return mask


def variable_band_filter(image, low_cutoff, high_cutoff):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape

    # Llama a la función compilada con Numba para calcular la máscara
    mask = calculate_mask(rows, cols, low_cutoff, high_cutoff)

    f_filtered = f_shift * mask

    f_inv_shift = np.fft.ifftshift(f_filtered)
    img_filtered = np.fft.ifft2(f_inv_shift)
    img_filtered = np.abs(img_filtered)

    return img_filtered.astype(np.uint16)