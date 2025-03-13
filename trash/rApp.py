#!/usr/bin/env python3
"""
production_pipeline.py

A production-level script to capture frames from the PiCamera2, process them
for 3D transformations, and visualize or store the results. Uses multithreading
to capture and process frames concurrently.
"""

import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List, Tuple

import numpy as np
import cv2

# External Dependencies
# from picamera2 import Picamera2
# from libcamera import controls

# Local Modules (example imports, adjust as needed)
# from src.Calibration import calibrateCamera
# from src.Scanner import Scanner, Parameters


###############################################################################
# Global Configuration
###############################################################################
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Camera Controls
FRAME_RATE = 56.0
ANALOGUE_GAIN = 1.0
EXPOSURE_TIME = 100000
OUTPUT_FILE = "dz_processed.npy"

# Queue / Thread Settings
QUEUE_MAX_SIZE = 400
NUM_WORKERS = 4
NUM_FRAMES_TO_CAPTURE = 200

# Geometry / Processing Parameters
TAN_30 = np.tan(np.pi / 6)

# Thread synchronization
stop_event = threading.Event()
frame_queue: Queue = Queue(maxsize=QUEUE_MAX_SIZE)
lock_processed = threading.Lock()


###############################################################################
# Camera Handler
###############################################################################
class CameraHandler:
    """
    Wrapper for initializing and interacting with the PiCamera2.
    This class can be extended to handle multiple configurations,
    calibrations, etc.
    """
    def __init__(self):
        """
        Initialize the camera using picamera2 library.
        """
        logger.info("Initializing camera...")
        # self.picam = Picamera2()
        # config = self.picam.create_video_configuration(
        #     main={"size": (2304, 1296), "format": "YUV420"}
        # )
        # self.picam.configure(config)
        # self.picam.set_controls({
        #     "FrameRate": FRAME_RATE,
        #     "AnalogueGain": ANALOGUE_GAIN,
        #     "ExposureTime": EXPOSURE_TIME,
        #     "AfMode": controls.AfModeEnum.Continuous
        # })
        # self.picam.start()

        # For demonstration without hardware, omit the real camera calls:
        self.picam = None

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the camera and return as a BGR numpy array.
        For demonstration, returns a dummy frame.
        """
        # Real capture:
        # yuv_frame = self.picam.capture_array()
        # frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        # return frame_bgr

        # Dummy synthetic frame for demonstration
        frame_dummy = np.random.randint(
            0, 255, (1296, 2304, 3), dtype=np.uint8
        )
        return frame_dummy

    def close(self):
        """
        Close or cleanup the camera resources if needed.
        """
        logger.info("Closing camera resources.")
        # self.picam.stop()


###############################################################################
# Frame Processor
###############################################################################
class FrameProcessor:
    """
    Encapsulates the logic needed to process camera frames using
    the Scanner class or any other transformation pipeline.
    """
    def __init__(self, ignore_camera: bool = True):
        """
        Initialize the frame processor with necessary parameters,
        e.g., load calibrations, homography matrix, etc.
        """
        logger.info("Initializing FrameProcessor...")
        # Replace with actual scanner initialization
        # self.scanner = Scanner(ignoreCamera=ignore_camera)
        # self.H_total = self.scanner.H_total
        self.scanner = None  
        self.H_total = np.eye(3)  # Identity for demonstration

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame and return the 'dz' data.
        The real logic will likely include advanced transformations.
        """
        try:
            # Simulate some complex transformations
            frame_rotated = np.rot90(frame_bgr, 1)

            # Example: You might have something like:
            # dx_data = self.scanner.processFrame(frame_rotated)
            # dx_data_points = self.scanner.getPoints(dx_data)
            # transformed_dx_data = cv2.perspectiveTransform(dx_data_points, self.H_total)
            # dz_p = transformed_dx_data[:, :, 1] / TAN_30
            # For demonstration, just return an array derived from the input
            h, w, _ = frame_rotated.shape
            dz_p = np.random.random((h, w))  # Dummy data

            return dz_p
        except Exception as ex:
            logger.error("Error processing frame: %s", ex)
            raise


###############################################################################
# Main Pipeline Functions
###############################################################################
def capture_frames(camera: CameraHandler, num_frames: int) -> None:
    """
    Capture frames from the camera and enqueue them for processing.
    This runs in a separate thread to avoid blocking.
    """
    logger.info("Camera warming up. Sleeping for 2s...")
    time.sleep(2)  # Let the camera stabilize

    for i in range(num_frames):
        if stop_event.is_set():
            logger.debug("Stop event set, halting frame capture.")
            break

        try:
            frame = camera.capture_frame()
            frame_queue.put((i, frame), block=True, timeout=0.5)
            logger.debug("Captured frame %d", i)
        except Exception as exc:
            logger.error("Error capturing frame %d: %s", i, exc)
            break

    logger.info("Finished capturing frames. Setting stop event.")
    stop_event.set()  # Indicate the capture phase has ended


def process_frames(processor: FrameProcessor, worker_id: int, results: List[Tuple[int, np.ndarray]]) -> None:
    """
    Continuously process frames from the queue until the capture phase
    is done and the queue is empty.
    """
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            i, frame = frame_queue.get(block=False)
        except Empty:
            time.sleep(0.1)
            continue

        start_time = time.time()
        try:
            dz_result = processor.process_frame(frame)
            with lock_processed:
                results.append((i, dz_result))
        except Exception as exc:
            logger.error("Worker %d failed to process frame %d: %s", worker_id, i, exc)
        finally:
            frame_queue.task_done()

        duration = time.time() - start_time
        logger.info("Worker %d processed frame %d in %.2f s", worker_id, i, duration)


def save_results(results: List[Tuple[int, np.ndarray]], output_path: str) -> None:
    """
    Sort the processed results by frame index, store the data into a
    single numpy array, and save to disk.
    """
    results.sort(key=lambda x: x[0])
    dz_array = np.array([res[1] for res in results])
    np.save(output_path, dz_array)
    logger.info("Saved %d frames of processed data to '%s'", len(results), output_path)


def visualize_results(data: np.ndarray) -> None:
    """
    Visualize the 3D results using matplotlib (optional in production).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # For 3D plotting
    
    # Suppose data has shape (num_frames, H, W)
    num_frames, height, width = data.shape
    y, x = np.meshgrid(
        np.arange(0, height), 
        np.arange(0, num_frames)
    )
    # For demonstration, we transpose to match typical x-y usage
    x = x.T
    y = y.T

    # A smoothing filter (optional)
    z_filtered = cv2.GaussianBlur(data, (11, 11), sigmaX=0, sigmaY=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z_filtered, cmap="viridis")
    plt.title("3D Processed Surface")
    plt.show()


###############################################################################
# Main Execution
###############################################################################
def main() -> None:
    camera = CameraHandler()
    processor = FrameProcessor(ignore_camera=True)
    results: List[Tuple[int, np.ndarray]] = []

    try:
        # Start capture thread
        capture_thread = threading.Thread(
            target=capture_frames,
            args=(camera, NUM_FRAMES_TO_CAPTURE),
            daemon=True
        )
        capture_thread.start()

        # Start a pool of worker threads
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for worker_id in range(NUM_WORKERS):
                executor.submit(process_frames, processor, worker_id, results)

        # Wait for all frames to be captured
        capture_thread.join()

        # Wait until all frames in the queue are processed
        frame_queue.join()

        # Save results
        save_results(results, OUTPUT_FILE)

        # Visualization (optional)
        # If you run headless, you may skip this or implement a separate
        # script to visualize saved data.
        # data_to_plot = np.load(OUTPUT_FILE)
        # visualize_results(data_to_plot)

    except KeyboardInterrupt:
        logger.warning("Received keyboard interrupt. Shutting down gracefully.")
        stop_event.set()
    except Exception as e:
        logger.exception("Unexpected error in main: %s", e)
    finally:
        # Ensure camera resources are released
        camera.close()
        logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
