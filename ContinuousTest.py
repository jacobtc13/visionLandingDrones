import numpy as np
import cv2
import io
from matplotlib import pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import picamera.array
import time


# Threshold values
bluehmn = 90
bluehmx = 130
redhmn = 160
redhmx = 179
smn = 20
smx = 255
vmn = 84
vmx = 255

kernel = np.ones((5,5),np.uint8)

def houghCircles(img):

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)

    # Apply thresholding
    bluehthresh = cv2.inRange(np.array(hue),np.array(bluehmn),np.array(bluehmx))
    redhthresh = cv2.inRange(np.array(hue),np.array(redhmn),np.array(redhmx))
    sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
    vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))

    # get final mask
    bluetracking = cv2.bitwise_and(bluehthresh,cv2.bitwise_and(sthresh,vthresh))
    redtracking = cv2.bitwise_and(redhthresh,cv2.bitwise_and(sthresh,vthresh))

    # some morpholigical filtering
    bluedilation = cv2.dilate(bluetracking,kernel,iterations = 1)
    blueclosing = cv2.morphologyEx(bluedilation, cv2.MORPH_CLOSE, kernel)
    blueclosing = cv2.GaussianBlur(blueclosing,(5,5),0)

    reddilation = cv2.dilate(redtracking,kernel,iterations = 1)
    redclosing = cv2.morphologyEx(reddilation, cv2.MORPH_CLOSE, kernel)
    redclosing = cv2.GaussianBlur(redclosing,(5,5),0)

    # detect circles
    bluecircles = cv2.HoughCircles(blueclosing,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
    redcircles = cv2.HoughCircles(redclosing,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)

    # draw cicles
    if bluecircles is not None:
        for i in bluecircles[0,:]:
            print("Blue Circle at: %s, %s Radius: %s" % (i[0], i[1], i[2]))
            cv2.circle(img,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(255,0,0),5)
            cv2.circle(img,(int(round(i[0])),int(round(i[1]))),2,(255,0,0),10)

    if redcircles is not None:
        for i in redcircles[0,:]:
            print("Red Circle at: %s, %s Radius: %s" % (i[0], i[1], i[2]))
            cv2.circle(img,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
            cv2.circle(img,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)

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

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):        
        # grab raw NumPy array
        image = frame.array
        
        # process image
        circles = houghCircles(image)

        # show the frame
        cv2.imshow("Output", circles)
        key = cv2.waitKey(1) & 0xFF
        
        # clear the stream for next frame
        rawCapture.truncate(0)
        
        # if the 'q' key is pressed, break from loop
        if key == ord("q"):
            break
            
            
if __name__ == "__main__":
    main()