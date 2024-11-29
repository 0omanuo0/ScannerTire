# # create a cell

# #%%python
# import numpy as np
# import zlib
# import timeit
# import blosc
# data = np.load("../frames.npy")
# # %%
# data[0]
# # get size in bytes of data[0]
# import sys
# d = data[0].copy()
# print(sys.getsizeof(data)/1024)
# print(sys.getsizeof(d)/1024)
# print(sys.getsizeof(np.packbits(d))/1024)
# t1 = timeit.default_timer()
# a = blosc.compress(np.packbits(d))
# t2 = timeit.default_timer()
# print("time: ", t2-t1)


import numpy as np
import threading
from queue import Queue, Empty
import cv2
import time
from src.Calibration import calibrateCamera
from src.Scanner import Scanner
from compute_dxData import compute_dxData_cython  # type: ignore

# Datos y configuración inicial
data = np.load("frames.npy")
queue = Queue(maxsize=200)
dz_processed = []
lock_processed = threading.Lock()

# Calibración y configuración inicial
H_total = calibrateCamera()
tan_30 = np.tan(np.pi / 6)
sc = Scanner(ignoreCamera=True)

# Evento para controlar la parada de los workers
stop_event = threading.Event()

# Función para generar frames
def frame_generator():
    for i, frame in enumerate(data):
        while not stop_event.is_set():
            try:
                queue.put([i, frame], block=True, timeout=0.1)
                print(f"Added frame {i} to queue, queue size: {queue.qsize()}")
                break
            except:
                print("Queue is full, waiting to add more frames...")
        time.sleep(0.0167)
    # Indicar que la generación de frames ha terminado
    stop_event.set()

# Función para que los workers procesen los frames
def frame_processor(worker_id):
    while not stop_event.is_set() or not queue.empty():
        try:
            i, frame = queue.get(block=False)
        except Empty:
            time.sleep(0.1)
            continue
        t1 = time.time()
        try:
            # Procesar el frame
            frame = np.rot90(frame, 2)
            dxData = sc.processFrame(frame)
            dxData = compute_dxData_cython(dxData)
            dxData_points = np.array(dxData, dtype=np.float32).reshape(-1, 1, 2)
            transformed_dxData = cv2.perspectiveTransform(dxData_points, H_total)
            dxP = transformed_dxData[:, :, 1]
            dzP = dxP / tan_30

            # Agregar resultado procesado a la lista
            with lock_processed:
                dz_processed.append([i, dzP.flatten()])
        except Exception as e:
            print(f"Error in worker {worker_id}: {e}")
        t2 = time.time()
        print(f"Worker {worker_id} processed frame {i} in {t2 - t1:.2f}s")



# Crear los hilos para los workers
workers = []
for i in range(4):
    worker = threading.Thread(target=frame_processor, args=(i,))
    worker.start()
    workers.append(worker)


# Crear el hilo para el generador
generator_thread = threading.Thread(target=frame_generator)
generator_thread.start()

# Esperar a que el generador termine
generator_thread.join()

# Esperar a que la cola se vacíe
while not queue.empty():
    time.sleep(0.1)

# Esperar a que los workers terminen
stop_event.set()
for worker in workers:
    worker.join()


# Imprimir el resultado final
print(f"Processed {len(dz_processed)} frames")