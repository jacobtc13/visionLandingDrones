from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

def main():
    # init the camera and get ref to raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):        
        # grab raw NumPy array
        image = frame.array
        
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        
        # clear the stream for next frame
        rawCapture.truncate(0)
        
        # if the 'q' key is pressed, break from loop
        if key == ord("q"):
            break
            
            
if __name__ == "__main__":
    main()