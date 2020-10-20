# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-04-18 09:50:54
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:24:33
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:03:02 2018

@author: gatv
"""

import cv2
import numpy as np
import time
from os.path import expanduser
home = expanduser("~")


cap = cv2.VideoCapture(home+'/Vídeos/DJI_0024.MOV')
ret, frame1 = cap.read()
#frame1 = cv2.resize(frame1,(0,0), fx=0.5, fy=0.5,interpolation = cv2.INTER_CUBIC)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
counter = 0
frame_rate = 5


scale = 1
delta = 0
ddepth = cv2.CV_64F


# 30 FPS
start_time = time.time()
while(ret):
    ret, frame2 = cap.read()
    counter +=1
#    frame2 = cv2.resize(frame2,(0,0), fx=0.5, fy=0.5,interpolation = cv2.INTER_CUBIC)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    if(counter%frame_rate == 0):

        #reduce de 20 a 2 fps
        #flow	=cv.calcOpticalFlowFarneback(prev,next,flow,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,flags	)
    
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 12, 3, 5, 1.2, 0)
#        flow = flow*10
#        copia = frame2.copy()
#        print(frame2.shape[0])
#        for y in range(0,frame2.shape[1],5):
#            for x in range(0,frame2.shape[0],5):
#                x_f = int(round((x+flow[x,y,0]),0))
#                y_f = int(round((y+flow[x,y,1]),0))
#                cv2.line(copia,(y,x),(y_f,x_f),(255,0,0),3)
##                cv2.circle(copia,(y,x),1,(0,255,0),-1)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        #SOBEL
        gray_bgr = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        test = gray_bgr > 80
        int_test = test.astype('uint8')
        int_test = int_test*255
        # Gradient-X
        grad_x = cv2.Sobel(gray_bgr,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
        #grad_x = cv2.Scharr(gray,ddepth,1,0)
        
        # Gradient-Y
        grad_y = cv2.Sobel(gray_bgr,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
        #grad_y = cv2.Scharr(gray,ddepth,0,1)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
        dst = cv2.add(abs_grad_x,abs_grad_y)
        cv2.namedWindow('optical_flow', cv2.WINDOW_NORMAL)
        cv2.imshow('optical_flow',int_test)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame',frame2)
        print(str(frame_rate/(time.time() - start_time))+' frames per second')
        start_time = time.time()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
cap.release()
cv2.destroyAllWindows()
