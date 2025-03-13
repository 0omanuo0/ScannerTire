from picamera2 import Picamera2

def get_available_resolutions():
    picam2 = Picamera2()
    camera_modes = picam2.sensor_modes  # Obtiene los modos soportados por el sensor
    print(camera_modes) 
    print("Resoluciones disponibles:")
    resolutions = []
    for mode in camera_modes:
        resolution = mode['size']
        resolutions.append(resolution)
        print(f"Resolution: {resolution[0]}x{resolution[1]}")

    return resolutions

# Llama a la función
resolutions = get_available_resolutions()