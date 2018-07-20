import numpy as np
import cv2
from matplotlib import pyplot as plt
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
	time1 = time.time()
    # setup AKAZE alg
	akaze = AKAZE_create(descriptor_type=cv2.DESCRIPTOR_MLDB)
	#akaze = Akaze_create(descriptor_type=cv2.DESCRIPTOR_KAZE)
	
	kp, des = akaze.detectAndCompute(img, None);
	print(time.time() - time1)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    cv2.imwrite('akazeShapes.png',img2)

def BriskFindFeatures(img)
	time1 = time.time()
    # setup BRISK alg
	brisk = BRISK_create()
	
	kp, des = brisk.detectAndCompute(img, None);
	print(time.time() - time1)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    cv2.imwrite('briskShapes.png',img2)

def OrbFindFeatures(img):
    print('Starting Orb Detection')
    time1 = time.time()
    # setup ORB alg
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    print('Orb Processing Time: %ss' % (time.time()-time1))
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    cv2.imwrite('orbShapes.png',img2)

def FastFindFeatures(img)
	time1 = time.time()
	# setup FAST alg
	fast = cv2.FastFeatureDetector_create()
	# disable nonmaxSuppression
	# fast.setNonmaxSuppression(0)
	kp = fast.detect(img,None)
	img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
	print(time.time() - time1)
	cv2.imwrite('fastShapes.png',img2)
	
def main():
    #cam = CameraCapture()
    #img = cam.capture()
    print('Loading Image')
    time0 = time.time()
    img = cv2.imread("shapes.jpg")
    print('Load Time: %ss' % (time.time()-time0))
    OrbFindFeatures(img)

    print('Total Time: %ss' % (time.time()-time0))

if __name__ == "__main__":
    main()
