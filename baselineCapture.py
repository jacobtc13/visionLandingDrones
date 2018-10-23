import numpy as np
import cv2
import io
import os
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import picamera.array
import time


class CameraCapture:
    def __init__(self, resx, resy):
        self.camera = PiCamera()
        self.camera.resolution = (resx,resy)
        self.camera.start_preview()
        #camera.color_effects = (128,128) #grayscale - needs to be tested to see if it improves anything
        self.camera.framerate = 32 # does this change anything?
        
        # camera warmup
        time.sleep(0.1)


    def capture(self):
        rawCapture = PiRGBArray(self.camera, size=(320,240))
        for foo in self.camera.capture_continuous(rawCapture,format="bgr"):  #the output is an RGB Array
            break
        image = rawCapture.array
        rawCapture.seek(0)
        rawCapture.truncate()
        return image


def main():
    mode = int(sys.argv[1])
    imagecounter = 0
    if(mode == 1):
        print('Capture mode: Normal')

        # setup camera 
        cam = CameraCapture(320, 240)
        
        raw_input('Setup camera 25cm from landing pad and press key')
        if not os.path.exists('25cm_0deg_0rot_NL'):
            os.makedirs('25cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_NL/25cm_0deg_0rot_NL_'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and press key')
        if not os.path.exists('50cm_0deg_0rot_NL'):
            os.makedirs('50cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_NL/50cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad and press key')
        if not os.path.exists('75cm_0deg_0rot_NL'):
            os.makedirs('75cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_NL/75cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad and press key')
        if not os.path.exists('150cm_0deg_0rot_NL'):
            os.makedirs('150cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_NL/150cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0

        ################

        raw_input('Setup camera 50cm from landing pad and 15 deg from ground and press key')
        if not os.path.exists('50cm_15deg_0rot_NL'):
            os.makedirs('50cm_15deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_15deg_0rot_NL/50cm_15deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and 30 deg from ground and press key')
        if not os.path.exists('50cm_30deg_0rot_NL'):
            os.makedirs('50cm_30deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_30deg_0rot_NL/50cm_30deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and 45 deg from ground and press key')
        if not os.path.exists('50cm_45deg_0rot_NL'):
            os.makedirs('50cm_45deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_45deg_0rot_NL/50cm_45deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0

        ################
        
        raw_input('Setup camera 50cm from landing pad, rotate landing pad 45 degrees and press key')
        if not os.path.exists('50cm_0deg_45rot_NL'):
            os.makedirs('50cm_0deg_45rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_45rot_NL/50cm_0deg_45rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
            
        raw_input('Setup camera 50cm from landing pad, rotate landing pad 90 degrees and press key')
        if not os.path.exists('50cm_0deg_90rot_NL'):
            os.makedirs('50cm_0deg_90rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_90rot_NL/50cm_0deg_90rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
            
        raw_input('Setup camera 50cm from landing pad, rotate landing pad 135 degrees and press key')
        if not os.path.exists('50cm_0deg_135rot_NL'):
            os.makedirs('50cm_0deg_135rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_135rot_NL/50cm_0deg_135rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
            
        raw_input('Setup camera 50cm from landing pad, rotate landing pad 180 degrees and press key')
        if not os.path.exists('50cm_0deg_180rot_NL'):
            os.makedirs('50cm_0deg_180rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_180rot_NL/50cm_0deg_180rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0


        raw_input('Setup camera 50cm from landing pad, camera 15 deg from ground, rotate object 45 degrees and press key')
        if not os.path.exists('50cm_15deg_45rot_NL'):
            os.makedirs('50cm_15deg_45rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_15deg_45rot_NL/50cm_15deg_45rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
            
        raw_input('Setup camera 50cm from landing pad, camera 30 deg from ground, rotate object 45 degrees and press key')
        if not os.path.exists('50cm_30deg_45rot_NL'):
            os.makedirs('50cm_30deg_45rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_30deg_45rot_NL/50cm_30deg_45rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
            
        raw_input('Setup camera 50cm from landing pad, camera 45 deg from ground, rotate object 45 degrees and press key')
        if not os.path.exists('50cm_45deg_45rot_NL'):
            os.makedirs('50cm_45deg_45rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_45deg_45rot_NL/50cm_45deg_45rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0


    elif(mode == 2):
        print('Capture mode: Resolutions')
        # test 640p
        cam = CameraCapture(640, 480)

        raw_input('Setup camera 25cm from landing pad and press key')
        if not os.path.exists('25cm_0deg_0rot_NL'):
            os.makedirs('25cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_NL/25cm_0deg_0rot_NL_'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and press key')
        if not os.path.exists('50cm_0deg_0rot_NL'):
            os.makedirs('50cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_NL/50cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad and press key')
        if not os.path.exists('75cm_0deg_0rot_NL'):
            os.makedirs('75cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_NL/75cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad and press key')
        if not os.path.exists('150cm_0deg_0rot_NL'):
            os.makedirs('150cm_0deg_0rot_NL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_NL/150cm_0deg_0rot_NL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0

        cam.close()

        # test 1080p
        cam = CameraCapture(1080, 800)
        
        raw_input('Setup camera 25cm from landing pad and press key')
        if not os.path.exists('25cm_0deg_0rot_NL_1080'):
            os.makedirs('25cm_0deg_0rot_NL_1080')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_NL_1080/25cm_0deg_0rot_NL_1080'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and press key')
        if not os.path.exists('50cm_0deg_0rot_NL_1080'):
            os.makedirs('50cm_0deg_0rot_NL_1080')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_NL_1080/50cm_0deg_0rot_NL_1080'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad and press key')
        if not os.path.exists('75cm_0deg_0rot_NL_1080'):
            os.makedirs('75cm_0deg_0rot_NL_1080')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_NL_1080/75cm_0deg_0rot_NL_1080'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad and press key')
        if not os.path.exists('150cm_0deg_0rot_NL_1080'):
            os.makedirs('150cm_0deg_0rot_NL_1080')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_NL_1080/150cm_0deg_0rot_NL_1080'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0

        cam.close()


    elif(mode == 3):
        print('Capture mode: NoIR')
         # setup camera 
        cam = CameraCapture(320, 240)
        
        raw_input('Setup camera 25cm from landing pad and press key')
        if not os.path.exists('25cm_0deg_0rot_NL_NoIR'):
            os.makedirs('25cm_0deg_0rot_NL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_NL_NoIR/25cm_0deg_0rot_NL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad and press key')
        if not os.path.exists('50cm_0deg_0rot_NL_NoIR'):
            os.makedirs('50cm_0deg_0rot_NL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_NL_NoIR/50cm_0deg_0rot_NL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad and press key')
        if not os.path.exists('75cm_0deg_0rot_NL_NoIR'):
            os.makedirs('75cm_0deg_0rot_NL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_NL_NoIR/75cm_0deg_0rot_NL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad and press key')
        if not os.path.exists('150cm_0deg_0rot_NL_NoIR'):
            os.makedirs('150cm_0deg_0rot_NL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_NL_NoIR/150cm_0deg_0rot_NL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0

        #######

        raw_input('Setup camera 25cm from landing pad with lights off and press key')
        if not os.path.exists('25cm_0deg_0rot_LL_NoIR'):
            os.makedirs('25cm_0deg_0rot_LL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_LL_NoIR/25cm_0deg_0rot_LL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad with lights off and press key')
        if not os.path.exists('50cm_0deg_0rot_LL_NoIR'):
            os.makedirs('50cm_0deg_0rot_LL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_LL_NoIR/50cm_0deg_0rot_LL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad with lights off and press key')
        if not os.path.exists('75cm_0deg_0rot_LL_NoIR'):
            os.makedirs('75cm_0deg_0rot_LL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_LL_NoIR/75cm_0deg_0rot_LL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad with lights off and press key')
        if not os.path.exists('150cm_0deg_0rot_LL_NoIR'):
            os.makedirs('150cm_0deg_0rot_LL_NoIR')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_LL_NoIR/150cm_0deg_0rot_LL_NoIR'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0


    elif(mode == 4):
        print('Capture mode: No lights normal camera')
        raw_input('Setup camera 25cm from landing pad with lights off and press key')
        if not os.path.exists('25cm_0deg_0rot_LL'):
            os.makedirs('25cm_0deg_0rot_LL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '25cm_0deg_0rot_LL/25cm_0deg_0rot_LL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 50cm from landing pad with lights off and press key')
        if not os.path.exists('50cm_0deg_0rot_LL'):
            os.makedirs('50cm_0deg_0rot_LL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '50cm_0deg_0rot_LL/50cm_0deg_0rot_LL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 75cm from landing pad with lights off and press key')
        if not os.path.exists('75cm_0deg_0rot_LL'):
            os.makedirs('75cm_0deg_0rot_LL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '75cm_0deg_0rot_LL/75cm_0deg_0rot_LL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        
        raw_input('Setup camera 150cm from landing pad with lights off and press key')
        if not os.path.exists('150cm_0deg_0rot_LL'):
            os.makedirs('150cm_0deg_0rot_LL')
        while(imagecounter != 100):
            imagecounter+=1
            img = cam.capture()
            writename = '150cm_0deg_0rot_LL/150cm_0deg_0rot_LL'+str(imagecounter)+'.png'
            cv2.imwrite(writename, img)
        imagecounter = 0
        

    print('Completed recording')
    


if __name__ == "__main__":
    main()

