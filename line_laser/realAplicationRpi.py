import numpy as np
import threading
from queue import Queue, Empty

import cv2
import sys
# print(cv2.ocl.haveOpenCL())  # Should return True
# cv2.ocl.setUseOpenCL(True)   # Enable OpenCL

import time

from concurrent.futures import ThreadPoolExecutor
from src.Calibration import calibrateCamera
from src.Scanner import Scanner, Parameters
from picamera2 import Picamera2
from libcamera import controls

import logging


# Camera setup
picam = Picamera2()
config = picam.create_video_configuration(main={"size": (2304, 1296), "format": "YUV420"})
picam.configure(config)
picam.set_controls({"FrameRate": 56.0, "AnalogueGain": 1.0, "ExposureTime": 100000, "AfMode": controls.AfModeEnum.Continuous})
picam.start()

# Initialization
frame_queue = Queue(maxsize=400)
dz_processed = []
lock_processed = threading.Lock()
scanner = Scanner(ignoreCamera=True)
stop_event = threading.Event()
tan_30 = np.tan(np.pi / 6)

times = []
queue_size_max = 0

# Function to capture frames from the camera
def frame_generator(num_frames=200):
    time.sleep(2)  # Allow the camera to stabilize
    
    time_frames = []
    t1 = time.time()
    for i in range(num_frames):
        try:
            
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

            if frame is not None:
                frame_queue.put((i, frame), block=True, timeout=0.1)
                frame_queue.put((i + num_frames, frame), block=True, timeout=0.1)
                frame_queue.put((i + num_frames*2, frame), block=True, timeout=0.1)
            else:
                logging.warning(f"Frame {i} is None, ignoring...")
                
            t2 = time.time()
            # print fps
            
            fps = 1 / (t2 - t1)
            t1 = t2
            time_frames.append(fps)
            fps_mean = np.mean(time_frames[:-5]) if len(time_frames) > 5 else np.mean(time_frames)
            log_message = f"\rFrame {i:<5} | FPS: {fps_mean:8.4f} | Queue Size: {frame_queue.qsize():<5}"
    
            # Imprimir sin salto de línea y forzar la actualización
            sys.stdout.write(log_message)
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"Error capturing frame {i}: {e}")
    print("")
    stop_event.set()  # Indicate that frame generation has finished



# Function to process frames
def frame_processor(worker_id):
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            i, frame = frame_queue.get(block=False)
        except Empty:
            time.sleep(0.1)
            continue

        global queue_size_max
        t_start = time.time()
        try:
            frame = np.rot90(frame, 1)
            dx_data = scanner.processFrame(frame)
            dx_data = np.rot90(dx_data, 3)
            dx_data_points = scanner.getPoints(dx_data)
            transformed_dx_data = cv2.perspectiveTransform(dx_data_points, scanner.H_total)
            dz_p = transformed_dx_data[:, :, 1] / tan_30

            with lock_processed:
                t_end = time.time()
                times.append(t_end - t_start)
                queue_size_max = max(queue_size_max, frame_queue.qsize())
        
                dz_processed.append((i, dz_p))
        except Exception as e:
            logging.error(f"Error in worker {worker_id} processing frame {i}: {e}")

    
        # logging.info(f"Worker {worker_id} processed frame {i} in {time.time() - t_start:.4f}s, queue size: {frame_queue.qsize()}")

        frame_queue.task_done()  # Mark as processed


# Main execution function
def main():
    num_workers = 3  # Number of processing threads
    num_frames = 200  # Total number of frames

    # Launch parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start frame generator
        generator_thread = threading.Thread(target=frame_generator, args=(num_frames,))
        generator_thread.start()

        # Start worker threads
        for i in range(num_workers):
            executor.submit(frame_processor, i)

        # Wait for completion
        generator_thread.join()
        frame_queue.join()  # Wait until the queue is fully emptied

    # Sort and save processed data
    dz_processed.sort(key=lambda x: x[0])
    dz_processed_np = np.array([x[1] for x in dz_processed])
    np.save("dz_processed.npy", dz_processed_np)
    
    # Log processing times
    logging.info(f"=====================RESULTS=====================")
    logging.info(f"Average processing time: {np.mean(times):.4f} s, std: {np.std(times):.4f} | mean ({num_workers} workers): {np.mean(times) / num_workers:.4f} s")
    logging.info(f"90th percentile processing time: {np.percentile(times, 10):.4f} s      | by worker:        {np.percentile(times, 90) / num_workers:.4f} s")
    logging.info(f"Maximum processing time: {np.max(times):.4f} s")
    logging.info(f"Minimum processing time: {np.min(times):.4f} s")
    logging.info(f"Maximum queue size: {queue_size_max}")
    logging.info(f"Processed {len(dz_processed)} frames and saved them in 'dz_processed.npy'")
    logging.info(f"=====================RESULTS=====================")
    
    # Generate visualization
    plot_results(dz_processed_np)

# Function to plot results in 3D
def plot_results(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    

    x_shape = len(data[0])
    y_shape = len(data)
    x, y = np.meshgrid(np.arange(0, x_shape), np.arange(0, y_shape))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    z_filtered = cv2.GaussianBlur(data, (11, 11), sigmaX=0, sigmaY=0)
    
    ax.plot_surface(x, y, z_filtered, cmap="viridis")
    plt.show()

# Run the program
if __name__ == "__main__":
    main()
