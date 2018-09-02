import numpy as np
import cv2
import io
from matplotlib import pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import picamera.array
import time

algnum = 0
numberoftests = 1
debugmode = false
filterimage = true

algorithms = { 0 : KazeFindFeatures,
               1 : BriskFindFeatures,
               2 : OrbFindFeatures,
               3 : FastFindFeatures,
               4 : HoughCircles,
               5 : LaplacianEdgeDetection,
               6 : SobelXEdgeDetection,
               7 : SobelYEdgeDetection,
               8 : CannyEdgeDetection
}

class CameraCapture:
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = (320,240)
        self.camera.start_preview()
        #camera.color_effects = (128,128) #grayscale - needs to be tested to see if it improves anything
        self.camera.framerate = 32
        
        # camera warmupq
        time.sleep(0.1)


    def capture(self):
        # capture frames from camera
        #for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
        rawCapture = PiRGBArray(self.camera, size=(320,240))
        #self.camera.capture(rawCapture, 'bgr')
        for foo in self.camera.capture_continuous(rawCapture,format="bgr"):  #the output is an RGB Array
            break
        #cv2.imshow('test1', rawCapture.array)- for debugging
        image = rawCapture.array
        rawCapture.seek(0)
        rawCapture.truncate()
        return image


def KazeFindFeatures(img, fimg):
    #print('Starting kaze Detection')
    # setup AKAZE alg
    kaze = cv2.KAZE_create()
    #akaze = cv2.AKAZE_create(descriptor_type=cv2.DESCRIPTOR_MLDB)
    #akaze = Akaze_create(descriptor_type=cv2.DESCRIPTOR_KAZE)
    kp, des = kaze.detectAndCompute(fimg, None)
    print('Akaze Processing Time: %ss' % (time.time()-time1))
    img = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    cv2.imwrite('kazeShapes.png',img)


def BriskFindFeatures(img, fimg):
    #print('Starting Brisk Detection')
    #setup BRISK alg
    brisk = cv2.BRISK_create()
    kp, des = brisk.detectAndCompute(fimg, None)
    print('Brisk Processing Time: %ss' % (time.time()-time1))
    img = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    #cv2.imwrite('briskShapes.png',img)


def OrbFindFeatures(img, fimg):
    #print('Starting Orb Detection')
    # setup ORB alg
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
    kp = orb.detect(fimg, None)
    kp, des = orb.compute(img, kp)
    print('Orb Processing Time: %ss' % (time.time()-time1))
    img = cv2.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
    #cv2.imwrite('orbShapes.png',img)


def FastFindFeatures(img, fimg):
    #print('Starting Fast Detection')
    # setup FAST alg
    fast = cv2.FastFeatureDetector_create()
    # disable nonmaxSuppression
    # fast.setNonmaxSuppression(0)
    kp = fast.detect(fimg,None)
    print('Fast Processing Time: %ss' % (time.time()-time1))
    img = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    #cv2.imwrite('fastShapes.png',img)

def HoughCircles(img, fimg):
    #print('Starting Hough Circles')
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(fimg,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
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
    #cv2.imwrite('houghCircles.png',img)

def LaplacianEdgeDetection(img, fimg):
    laplacian = cv2.Laplacian(fimg,cv2.CV_64F)

    #draw filtered image
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

def SobelXEdgeDetection(img, fimg):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    #draw filtered image
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')

def SobelYEdgeDetection(img, fimg):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    #draw filtered image
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


def CannyEdgeDetection(img, fimg)
    # Create a matrix of the same type and size as src (for dst)
    #dst.create( src.size(), src.type() )

    # Create a window
    cv2.namedWindow( 'Edge map' )

    # Create a Trackbar for user to enter threshold
    cv2.createTrackbar( 'thresh', 'Edge map', 1, 100, nothing)

    cv2.blur(fimg, detected_edges, Size(3,3))

    thres = cv2.getTrackbarPos('thresh','Edge map')
    
    cv2.Canny(detected_edges, detected_edges, thres, thres*3, 3)

    #show image here

def Grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def FilterImage(img):
    kernel = np.ones((5,5),np.uint8)
    
    def nothing(x):
        pass

    #converting to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)

    # Apply thresholding
    hthresh = cv2.inRange(np.array(hue),50,60) # include Hue min and max values here between 0-179
    sthresh = cv2.inRange(np.array(sat),50,60) # include Sat min and max values here between 0-255
    vthresh = cv2.inRange(np.array(val),50,60) # include Vlaue min and max values here between 0-255

    # AND h s and v
    tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

    # Some morpholigical filtering
    dilation = cv2.dilate(tracking,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    closing = cv2.GaussianBlur(closing,(5,5),0)
    return closing


def DebugFilters(cam):
    kernel = np.ones((5,5),np.uint8)
    cam = CameraCapture()
    #print('pass')
    
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

        #Show the result in frames
        cv2.imshow('HueComp',hthresh)
        cv2.imshow('SatComp',sthresh)
        cv2.imshow('ValComp',vthresh)
        cv2.imshow('closing',closing)
        cv2.imshow('tracking',img)

def main():
    # create camera instance
    cam = CameraCapture()

    if(debugmode):
        # For testing HSV ranges
        DebugFilters(cam)
    else:
        totaltime = 0.0
        testsremaining = numberoftests
        while(testsremaining != 0):   
            img = cam.capture()
            fimg = None

            timestart = time.time()
            # Filter the image if filtering is turned on, else use unfiltered
            # image for drawing on and display
            if(filterimage):
                fimg = FilterImage(img)
                algorithms[algnum](img, fimg)
            else:
                algorithms[algnum](img, img)
                
            totaltime += (time.time() - timestart)
            testsremaining -= 1
        print('Total Time: %ss' % totaltime)
        print('Average time taken per frame: %ss' % (totaltime/numberoftests))

    # cleanup after running tests
    cv2.destroyAllWindows()
    
    


if __name__ == "__main__":
    main()

