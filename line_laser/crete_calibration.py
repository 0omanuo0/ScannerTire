from picamera2 import Picamera2
from libcamera import controls
import cv2


picam = Picamera2()
config = picam.create_video_configuration(
        main={"size": (2304, 1296), "format": "YUV420"})
# set camera configuration iso, shutter speed...
picam.configure(config)

# success = picam.autofocus_cycle()
# if(not success):
#     print("Autofocus failed")
#     exit(1) 
picam.set_controls({"FrameRate": 56.0,"AnalogueGain": 4.0, "ExposureTime": 100000, "AfMode": controls.AfModeEnum.Continuous})
picam.start()

counter = 0

iso = 4.0
shutter_speed = 100000

# set slider opencv
cv2.namedWindow('Data', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Data', 1000, 500)
cv2.createTrackbar('ISO', 'Data', 0, 15, lambda x: None)
cv2.createTrackbar('Shutter Speed', 'Data', 1000, 100000, lambda x: None)
# cv2.createTrackbar('ColorFilter1', 'Data', 0, 255, lambda x: None)
# cv2.createTrackbar('ColorFilter2', 'Data', 0, 255, lambda x: None)
cv2.setTrackbarPos('ISO', 'Data', 4)
cv2.setTrackbarPos('Shutter Speed', 'Data', 10000)

while True:
    
    iso = cv2.getTrackbarPos('ISO', 'Data')
    shutter_speed = cv2.getTrackbarPos('Shutter Speed', 'Data')
    # color_filter1 = cv2.getTrackbarPos('ColorFilter1', 'Data')
    # color_filter2 = cv2.getTrackbarPos('ColorFilter2', 'Data')

    picam.set_controls({"AnalogueGain": iso, "ExposureTime": shutter_speed})

    f = picam.capture_array()
    if f is None:
        continue

    f = cv2.cvtColor(f, cv2.COLOR_YUV2BGR_I420)
    
    cv2.imshow('Data', f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"frame_{counter}.jpg", f)
        counter += 1
        print("Frame {} saved".format(counter))

picam.stop()
