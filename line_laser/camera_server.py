import gpiod
import numpy as np
import threading
from queue import Queue, Empty
import flask
import uuid
import os

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

# Pines GPIO
channelPinA = 6
channelPinB = 19
channelPinZ = 26
laserPin = 14

value = 0
valuez = 0
lock = threading.Lock()

chip = gpiod.Chip('gpiochip0')
lineA = chip.get_line(channelPinA)
lineB = chip.get_line(channelPinB)
lineZ = chip.get_line(channelPinZ)
laser = chip.get_line(laserPin)

lineA.request(consumer="encoderA", type=gpiod.LINE_REQ_EV_RISING_EDGE)
lineB.request(consumer="encoderB", type=gpiod.LINE_REQ_DIR_IN)
lineZ.request(consumer="encoderZ", type=gpiod.LINE_REQ_DIR_IN)
laser.request(consumer="laser", type=gpiod.LINE_REQ_DIR_OUT, default_val=0)


laser.set_value(1)  # Enciende el láser

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


tire_radius = 2044 # truck circumference in steps (in mm =)
n_frames = 100
# the camera is 1/2 faster than the tire so if needs to make 100 frames in 1 second,
# the camera is going to do 2 rounds to get all the frames (one for odd and one for even)
# but to be sure we need to asume that is half of the speed (the multiplier)
rotation_speed_multiplier = 2 
camera_speed = 50 # camera speed in fps
# 40rps in encoder or 1rps in tire
rps = 1

# Function to capture frames from the camera
def frame_generator(num_frames:int):
    global value, valuez, laser
    laser.set_value(1)
    time.sleep(2)  # Allow the camera to stabilize
    time_frames = []
    t1 = time.time()
    
    prevStateChannelZ = lineZ.get_value()
    prev_count = 0
    
    
    # how many revolutions are going to be made to take the hole scan
    rev_times = (((n_frames * rps) / camera_speed) * rotation_speed_multiplier)
    
    # Calculate the number of steps per frame, based on the tire radius and camera speed and a factor to be sure
    steps_per_frame = (tire_radius / n_frames) * rev_times

    
    # how many revolutions are taken
    revs = -1
    i = 0

    while True:        
        # Wait for the encoder to trigger
        if lineA.event_wait():
            event = lineA.event_read()
            currentStateChannelZ = lineZ.get_value()
            if lineB.get_value() == 1:
                value += 1
                if prevStateChannelZ != currentStateChannelZ:
                    prevStateChannelZ = currentStateChannelZ
                    valuez += 1
            else:
                value -= 1
                if prevStateChannelZ != currentStateChannelZ:
                    valuez -= 1
                    prevStateChannelZ = currentStateChannelZ

            with lock:
                log_message = f" | V = {value}    Z = {int(valuez/2)}"
                
        # Check if has taken a full revolution
        if value % tire_radius == 0:
            revs += 1
        
        # case 0 and on each rev, ex: if steps_per_frame is 64 and rev_times is 4 and tire_radius is 1024:
        # cases 0, 1032, 2056 ...
        # if value == 0 or value >= tire_radius + (steps_per_frame / rev_times) * revs:
        # Check if the encoder has moved enough to capture a frame 
        # if has taken enough steps, if is 0 or if has taken a full revolution + the offset
        if value - prev_count >= steps_per_frame \
            or value == 0 \
            or value >= tire_radius + (steps_per_frame / rev_times) * revs:
                
            # update the value
            prev_count = value
            
            # ends the loop if has taken enough frames
            if len(time_frames) >= n_frames:
                # reset count 
                value = 0
                valuez = 0
                break
            
            i += 1
            
            try:
                frame = picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                if frame is not None:
                    # Put the frame in the queue with the corresponding index
                    # the index is calculated as the number of frames taken interlacing the frames, 
                    # each rev only takes each n frames, 0, 4, 8; 1, 5, 9; 2, 6, 10; 3, 7, 11...
                    frame_queue.put(((i*rev_times - rev_times + revs), frame), block=True, timeout=0.1)
                else:
                    logging.warning(f"Frame {(i*rev_times - rev_times + revs)} is None, ignoring...")
                    
                t2 = time.time()
                fps = 1 / (t2 - t1)
                t1 = t2
                time_frames.append(fps)
                fps_mean = np.mean(time_frames[:-5]) if len(time_frames) > 5 else np.mean(time_frames)
                log_message += f"\rFrame {i:<5} | FPS: {fps_mean:8.4f} | Queue Size: {frame_queue.qsize():<5}"
        
                # Imprimir sin salto de línea y forzar la actualización
                sys.stdout.write(log_message)
                sys.stdout.flush()
            except Exception as e:
                logging.error(f"Error capturing frame {i}: {e}")
    print("")
    stop_event.set()  # Indicate that frame generation has finished
    laser.set_value(0)


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

is_scanning = False

# Main execution function
def scan(id:str):
    
    global is_scanning
    
    is_scanning = True
    
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
    np.save(f"scans/dz_processed-{id}.npy", dz_processed_np)
    
    # Log processing times
    logging.info(f"=====================RESULTS=====================")
    logging.info(f"Average processing time: {np.mean(times):.4f} s, std: {np.std(times):.4f} | mean ({num_workers} workers): {np.mean(times) / num_workers:.4f} s")
    logging.info(f"90th percentile processing time: {np.percentile(times, 10):.4f} s      | by worker:        {np.percentile(times, 90) / num_workers:.4f} s")
    logging.info(f"Maximum processing time: {np.max(times):.4f} s")
    logging.info(f"Minimum processing time: {np.min(times):.4f} s")
    logging.info(f"Maximum queue size: {queue_size_max}")
    logging.info(f"Processed {len(dz_processed)} frames and saved them in 'dz_processed.npy'")
    logging.info(f"=====================RESULTS=====================")
    
    is_scanning = False
    # Generate visualization
    # plot_results(dz_processed_np)

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
    

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = flask.Flask(__name__)
@app.route('/get/<id>', methods=['GET'])
def get(id):
    print(id)
    try:
        file_path = os.path.join(BASE_DIR, "scans", f"dz_processed-{id}.npy")
        print(file_path)
        return flask.send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        return flask.jsonify({"error": "File not found"}), 404
    
@app.route('/scan/', methods=['GET'])
def server_scan():
    # generate uuid4
    id = str(uuid.uuid4())
    # create directory
    try:
        os.makedirs("scans")
    except FileExistsError:
        pass
    
    # start scan
    logging.info(f"Starting scan with id {id}")
    if is_scanning:
        return flask.jsonify({"error": "Already scanning"}), 400
    else:
        th_s = threading.Thread(target=scan, args=(id,))
        th_s.start()
        return flask.jsonify({"id": id}), 200
        

# Run the program1
if __name__ == "__main__":
    try:
        time.sleep(1)
        laser.set_value(0)
        app.run(host="0.0.0.0", debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Saliendo...")
    finally:
        laser.set_value(0)  # Apaga el láser
        lineA.release()
        lineB.release()
        lineZ.release()
        laser.release()
