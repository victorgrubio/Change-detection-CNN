# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-04-23 11:36:11
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:24:40
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:48:49 2018

@author: visiona
"""

import numpy as np
import cv2

from os.path import expanduser

frame_list = []
home = expanduser("~")
cap = cv2.VideoCapture(home+'/Vídeos/DJI_0024.MOV')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (12,12),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.createBackgroundSubtractorMOG2(500,16,False)

counter = 0
while(ret):
    counter += 1
    ret,frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # calculate optical flow
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
#    good_new = p1[st==1]
#    good_old = p0[st==1]
##    h, status = cv2.findHomography(good_new, good_old)
##    im_dst = cv2.warpPerspective(frame,h,(frame_gray.shape[1],frame_gray.shape[0]))
##    # draw the tracks
#    points  = zip(good_new,good_old)
#    vectors = good_new - good_old
#    mean_x  = np.mean(vectors[0])
#    mean_y  = np.mean(vectors[1])
#    print()
#    print(mean_x)
#    print(mean_y)
#    print('-------')
#    for i,(new,old) in enumerate(points):
#        a,b = new.ravel()
#        c,d = old.ravel()
#        if(abs(vectors[0,i]-mean_x < 0.5)):
#            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#        mov_x = c-a
#        mov_y = d-b
#        print(str(mov_x))
#        print(str(mov_y))
#        print('-------')
#        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    if(counter%5 == 0):
#        fgmask = fgbg.apply(frame)
#        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#        img = cv2.add(frame,mask)
#        cv2.namedWindow('background', cv2.WINDOW_NORMAL)
#        cv2.imshow('background',fgmask)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
#    old_gray = frame_gray.copy()
#    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()