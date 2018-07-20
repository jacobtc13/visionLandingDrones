import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class CameraCapture:
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = (320,240)
        #camera.color_effects = (128,128) #grayscale
        self.camera.framerate = 32
        self.rawCapture = PiRGBArray(self.camera, size=(320,240))
        # camera warmupq
        time.sleep(0.1)


    def capture(self):
        # capture frames from camera
        #for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        	# grab the raw NumPy array representing the image, then initialize the timestamp
        	# and occupied/unoccupied text
        self.camera.capture(self.rawCapture, 'bgr')
        image = self.rawCapture.array
        return image


def AkazeFindFeatures(img):

    akaze = Akaze_create(descriptor_type=cv2.DESCRIPTOR_KAZE)


def OrbFindFeatures(img):
    time1 = time.time()
    # setup ORB alg
    orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, color(0,255,0),flags=0)
    print(time.time() - time1)
    cv2.imwrite('orbShapes.png',img)

def main():
    #cam = CameraCapture()
    #img = cam.capture()
    time0 = time.time()
    img = cv2.imread("shapes.jpg")

    OrbFindFeatures(img)



if __name__ == "__main__":
    main()
