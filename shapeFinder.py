import numpy as np
import cv2
import io
from matplotlib import pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import picamera.array
import time

class CameraCapture:
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = (320,240)
        #camera.color_effects = (128,128) #grayscale
        self.camera.framerate = 32
        
        # camera warmupq
        time.sleep(0.1)


    def capture(self):
        # capture frames from camera
        #for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
        self.rawCapture = PiRGBArray(self.camera, size=(320,240))
        self.camera.capture(self.rawCapture, 'bgr')
        image = self.rawCapture.array
        return image


def KazeFindFeatures(img):
    print('Starting kaze Detection')
    time1 = time.time()
    # setup AKAZE alg
    kaze = cv2.KAZE_create()
    #akaze = cv2.AKAZE_create(descriptor_type=cv2.DESCRIPTOR_MLDB)
    #akaze = Akaze_create(descriptor_type=cv2.DESCRIPTOR_KAZE)
    kp, des = kaze.detectAndCompute(img, None)
    print('Akaze Processing Time: %ss' % (time.time()-time1))
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    cv2.imwrite('kazeShapes.png',img2)


def BriskFindFeatures(img):
    print('Starting Brisk Detection')
    time1 = time.time()
    #setup BRISK alg
    brisk = cv2.BRISK_create()
    kp, des = brisk.detectAndCompute(img, None)
    print('Brisk Processing Time: %ss' % (time.time()-time1))
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


def FastFindFeatures(img):
    print('Starting Fast Detection')
    time1 = time.time()
    # setup FAST alg
    fast = cv2.FastFeatureDetector_create()
    # disable nonmaxSuppression
    # fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)
    print('Fast Processing Time: %ss' % (time.time()-time1))
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    cv2.imwrite('fastShapes.png',img2)


def ImageTest():
    kernel = np.ones((5,5),np.uint8)
    cam = CameraCapture()
    print('pass')
    
    def nothing(x):
        pass
    
    cv2.namedWindow('HueComp')
    cv2.namedWindow('SatComp')
    cv2.namedWindow('ValComp')
    cv2.namedWindow('closing')
    cv2.namedWindow('tracking')

    cv2.createTrackbar('hmin', 'HueComp',12,179,nothing)
    cv2.createTrackbar('hmax', 'HueComp',37,179,nothing)
    cv2.createTrackbar('smin', 'SatComp',96,255,nothing)
    cv2.createTrackbar('smax', 'SatComp',255,255,nothing)
    cv2.createTrackbar('vmin', 'ValComp',186,255,nothing)
    cv2.createTrackbar('vmax', 'ValComp',255,255,nothing)

    while(1):
        img = cam.capture()
        #converting to HSV
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hue,sat,val = cv2.split(hsv)

        # get info from track bar and appy to result
        hmn = cv2.getTrackbarPos('hmin','HueComp')
        hmx = cv2.getTrackbarPos('hmax','HueComp')
        smn = cv2.getTrackbarPos('smin','SatComp')
        smx = cv2.getTrackbarPos('smax','SatComp')
        vmn = cv2.getTrackbarPos('vmin','ValComp')
        vmx = cv2.getTrackbarPos('vmax','ValComp')

	# Apply thresholding
        hthresh = cv2.inRange(np.array(hue),np.array(hmn),np.array(hmx))
        sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
        vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))

	# AND h s and v
        tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

	# Some morpholigical filtering
        dilation = cv2.dilate(tracking,kernel,iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        closing = cv2.GaussianBlur(closing,(5,5),0)

	# Detect circles using HoughCircles
        circles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
	# circles = np.uint16(np.around(circles))


	#Draw Circles
        if circles is not None:
            for i in circles[0,:]:
                # If the ball is far, draw it in green
                if int(round(i[2])) < 30:
                    cv2.circle(img,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
                    cv2.circle(img,(int(round(i[0])),int(round(i[1]))),2,(0,255,0),10)
                # else draw it in red
                elif int(round(i[2])) > 35:
                    cv2.circle(img,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
                    cv2.circle(img,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)
                    buzz = 1

    #you can use the 'buzz' variable as a trigger to switch some GPIO lines on Rpi :)
    # print buzz
    # if buzz:
        # put your GPIO line here


    #Show the result in frames
                #cv2.imshow('HueComp',hthresh)
                #cv2.imshow('SatComp',sthresh)
                #cv2.imshow('ValComp',vthresh)
                #cv2.imshow('closing',closing)
                cv2.imshow('tracking',img)
                
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
        print('Complete')
        #time.sleep(20)
        #cv2.destroyAllWindows()

def main():
    cam = CameraCapture()
    img = cam.capture()
    cv2.imshow('test', img)
    #print('Loading Image')
    #time0 = time.time()
    #img = cv2.imread("shapes.jpg")
    #print('Load Time: %ss' % (time.time()-time0))
    #OrbFindFeatures(img)
    #FastFindFeatures(img)
    #BriskFindFeatures(img)
    #KazeFindFeatures(img)
    #ImageTest()
    print('Total Time: %ss' % (time.time()-time0))
    
    


if __name__ == "__main__":
    main()

