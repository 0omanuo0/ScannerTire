# Line Laser Scanner

This project implements a line laser scanner using OpenCV and Python. The scanner captures frames from a camera, processes them to detect a laser spot, calculates depth values, and communicates with an Arduino to control a stepper motor.

## Classes and Methods

### Parameters Class
The `Parameters` class is used to initialize the scanner's parameters.

**Attributes:**
- `fov` (float): Field of view.
- `theta` (float): Angle theta.
- `L` (float): Distance L.
- `resolution` (tuple[int, int]): Resolution of the camera, default is (1280, 720).

**Methods:**
- `__init__(self, fov: float, theta: float, L: float, resolution: tuple[int, int] = (1280, 720))`: Initializes the Parameters object.

### Scanner Class
The `Scanner` class represents a line laser scanner.

**Attributes:**
- `DEFAULT_PARAMETERS` (Parameters): Default parameters for the scanner.
- `Filters` (Enum): Enumeration of filters for the scanner.

**Methods:**
- `__init__(self, parameters: Parameters = DEFAULT_PARAMETERS, serial_port:str = None)`: Initializes the Scanner object, connects to the serial port, and sets up the camera.
- `__waitResponseSerial(self, timeout:int = 2)`: Waits for a response from the serial port with a timeout.
- `calculate_depth(self, image:np.ndarray) -> np.ndarray`: Calculates the depth value for each column with a white pixel (laser spot) in the image.
- `processFrame(self, frame: np.ndarray) -> np.ndarray`: Processes the frame to detect the laser spot.
- `getFrame(self)->np.ndarray`: Captures a frame from the camera.
- `getProcessedFrame(self) -> np.ndarray`: Gets a processed frame with the laser spot.
- `getDepthValues(self, frame) -> np.ndarray`: Gets the depth values for the laser spot in the frame.
- `sendSteps(self, steps: int)`: Sends the number of steps to the Arduino.
- `getSteps(self, filter:Filters = 0)->list[np.ndarray]`: Gets the number of steps from the Arduino and returns a list of processed frames.
- `__del__(self)`: Releases the camera and closes the serial port.

## Usage

1. Initialize the `Scanner` object with the desired parameters and serial port.
    ```python
    params = Parameters(fov=60, theta=0, L=1000, resolution=(480, 640))
    scanner = Scanner(parameters=params, serial_port='COM7')
    ```

2. Capture a frame and process it to detect the laser spot.
    ```python
    frame = scanner.getFrame()
    processed_frame = scanner.getProcessedFrame()
    ```

3. Calculate the depth values for the laser spot in the frame.
    ```python
    depth_values = scanner.getDepthValues(frame)
    ```

4. Send steps to the Arduino.
    ```python
    scanner.sendSteps(100)
    ```

5. Get frames from the Arduino and process them.
    ```python
    frames = scanner.getSteps(filter=Scanner.Filters.DEPTH_VALUES)
    ```

## Dependencies

- `cv2` (OpenCV)
- `numpy`
- `serial`

Make sure to install the required libraries before running the code:
```bash
pip install opencv-python numpy pyserial
```

# License

This project is licensed under the MIT License.