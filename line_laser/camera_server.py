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
from src.Scanner import Scanner
from picamera2 import Picamera2
from libcamera import controls
import math

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
picam.set_controls({"FrameRate": 56.0, "AnalogueGain": 7.0, "ExposureTime": 1000, "AfMode": controls.AfModeEnum.Continuous})
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


# take a picture and measure the ´bytes
frame = picam.capture_array()

tire_radius = 2044 # truck circumference in steps (in mm =)
# tire_radius = 1636 # 1413.7164 circumference perimeter, r=4.09090 => 1636steps
n_frames = 500
# the camera is 1/2 faster than the tire so if needs to make 100 frames in 1 second,
# the camera is going to do 2 rounds to get all the frames (one for odd and one for even)
# but to be sure we need to asume that is half of the speed (the multiplier)
rotation_speed_multiplier = 2
camera_speed = 50 # camera speed in fps
# 40rps in encoder or 1rps in tire
# rps = 0.25
rps=0.1

# Function to capture frames from the camera
def frame_generator(num_frames: int):
    global value, valuez, laser
    laser.set_value(1)
    time.sleep(2)  # Allow the camera to stabilize
    time_frames = []
    t1 = time.time()
    
    prevStateChannelZ = lineZ.get_value()
    prev_count = 0
    
    # how many revolutions are going to be made to take the hole scan
    rev_times = math.ceil(((n_frames * rps) / camera_speed) * rotation_speed_multiplier)
    
    # Calculate the number of steps per frame, based on the tire radius and camera speed and a factor to be sure
    steps_per_frame = int((tire_radius / n_frames) * rev_times)

    # how many revolutions are taken
    revs = 0
    frame_index = 0  # global frame index to prevent duplicate frame tags
    frames_per_rev = n_frames // rev_times

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
                    # Check if has taken a full revolution
                    # if value % tire_radius == 0:
                    #     revs -= 1


            # with lock:
            #     log_message = f" | V = {value}    Z = {int(valuez/2)}"
                

        # case 0 and on each rev, ex: if steps_per_frame is 64 and rev_times is 4 and tire_radius is 1024:
        # cases 0, 1032, 2056 ...
        # if value == 0 or value >= tire_radius + (steps_per_frame / rev_times) * revs:
        # Check if the encoder has moved enough to capture a frame 
        # if has taken enough steps, if is 0 or if has taken a full revolution + the offset
        #            or value == 0 \
        if value - prev_count >= steps_per_frame \
            or value >= tire_radius + (steps_per_frame / rev_times) * revs:
            
            # Check if has taken a full revolution
            if value % tire_radius == 0:
                revs += 1

            # update the value
            prev_count = value

            # ends the loop if has taken enough frames
            if len(time_frames) >= n_frames:
                # reset count 
                value = 0
                valuez = 0
                break

            try:
                frame = picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                # the tag is the frame index but with the phase
                # each rev only takes each n frames, 0, 4, 8; 1, 5, 9; 2, 6, 10; 3, 7, 11...
                frame_tag = ((frame_index % frames_per_rev) * rev_times) + revs
                frame_index += 1

                if frame is not None:
                    # Put the frame in the queue with the corresponding index
                    frame_queue.put((frame_tag, frame), block=True, timeout=0.1)
                else:
                    logging.warning(f"Frame {frame_tag} is None, ignoring...")

                t2 = time.time()
                fps = 1 / (t2 - t1)
                t1 = t2
                time_frames.append(fps)
                fps_mean = np.mean(time_frames[:-5]) if len(time_frames) > 5 else np.mean(time_frames)
                log_message = f"\rFrame {frame_tag:<5} | FPS: {fps_mean:8.4f} | Queue Size: {frame_queue.qsize():<5}"

                # Imprimir sin salto de línea y forzar la actualización
                sys.stdout.write(log_message)
                sys.stdout.flush()
            except Exception as e:
                logging.error(f"Error capturing frame {frame_index}: {e}")
                os._exit(1)

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
            # save frame as jpg
            cv2.imwrite(f"frames/frame_{i}.jpg", frame)
            frame = np.rot90(frame, 1)
            dx_data = scanner.processFrame(frame)
            dx_data = np.rot90(dx_data, 3)
            dx_data_points = scanner.getPoints(dx_data)
            transformed_dx_data = cv2.perspectiveTransform(dx_data_points, scanner.H_total)
            dz_p = transformed_dx_data[:, :, 1] / tan_30
            # dz_p = dz_p[200:1000, :]
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
    print(dz_processed_np.shape)
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
