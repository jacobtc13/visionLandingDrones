from picamera.array import PiRGBArray
from picamera import PiCamera
from matplotlib import pyplot as plt
import time
import cv2
import io
import numpy as np

def setupTrackbars():
    cv2.namedWindow('HueComp')
    cv2.namedWindow('SatComp')
    cv2.namedWindow('ValComp')
    cv2.namedWindow('closing')
    cv2.namedWindow('tracking')

    def nothing(x):
        pass

    cv2.createTrackbar('hmin', 'HueComp',12,179,nothing)
    cv2.createTrackbar('hmax', 'HueComp',37,179,nothing)
    cv2.createTrackbar('smin', 'SatComp',96,255,nothing)
    cv2.createTrackbar('smax', 'SatComp',255,255,nothing)
    cv2.createTrackbar('vmin', 'ValComp',186,255,nothing)
    cv2.createTrackbar('vmax', 'ValComp',255,255,nothing)

def houghCircles(img):
    kernel = np.ones((5,5),np.uint8)

    # converting to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)

    # get info from track bar and appy to result
    hmn = 20 #cv2.getTrackbarPos('hmin','HueComp')
    hmx = 160 #cv2.getTrackbarPos('hmax','HueComp')
    smn = 100 #cv2.getTrackbarPos('smin','SatComp')
    smx = 255 #cv2.getTrackbarPos('smax','SatComp')
    vmn = 186 #cv2.getTrackbarPos('vmin','ValComp')
    vmx = 255 #cv2.getTrackbarPos('vmax','ValComp')

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

    return img


def main():

    # init the camera and get ref to raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    #setupTrackbars()
    cv2.namedWindow("Raw")
    cv2.namedWindow("Circles")

    time0 = time.time()

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):        
        
        # grab raw NumPy array
        image = frame.array

        time1 = time.time()

        cv2.imshow("Raw", image)

        # process frame
        circles = houghCircles(image)

        # show the frame
        cv2.imshow("Circles", circles)
        key = cv2.waitKey(1) & 0xFF
        
        time2 = time.time()

        # clear the stream for next frame
        rawCapture.truncate(0)
        
        # if the 'q' key is pressed, break from loop
        if key == ord("q"):
            break

        print('Get Frame: %s, Process Frame: %s, Total: %s' % (time1-time0, time2-time1, time.time()-time0))
        time0 = time.time()
            
            
if __name__ == "__main__":
    main()