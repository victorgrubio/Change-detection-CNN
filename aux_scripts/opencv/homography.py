# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-05-25 12:16:39
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:24:43
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:48:49 2018

@author: visiona
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('/home/visiona/MEGA/Video/Pruebas Montegancedo/DJI_0001.MOV')
cap2 = cv2.VideoCapture('/home/visiona/MEGA/Video/Pruebas Montegancedo/DJI_0002.MOV')
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

while(1):

    ret, old_frame = cap.read()
    ret,frame = cap2.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    p0_b = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    p1_b, st_b, err_b = cv2.calcOpticalFlowPyrLK(frame_gray,old_gray, p0_b, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_new_b = p1_b[st_b==1]
    good_old_b = p0_b[st_b==1]


    h, status = cv2.findHomography(good_new, good_old)
    h_b, status_b = cv2.findHomography(good_new_b, good_old_b)

    print("type: ", type(good_new))
    print("shape: ", good_new.shape)


    im_dst = cv2.warpPerspective(frame,h,(frame_gray.shape[1],frame_gray.shape[0]))
    im_dst_b = cv2.warpPerspective(old_frame,h_b,(old_gray.shape[1],old_gray.shape[0]))

    im_dst_gray = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    im_dst_b_gray = cv2.cvtColor(im_dst_b, cv2.COLOR_BGR2GRAY)

    diff = abs(im_dst_gray-im_dst_b_gray)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',im_dst)
    cv2.namedWindow('background', cv2.WINDOW_NORMAL)
    cv2.imshow('background',old_frame)
    cv2.namedWindow('foreground', cv2.WINDOW_NORMAL)
    cv2.imshow('foreground',frame)
    cv2.namedWindow('diff_homography', cv2.WINDOW_NORMAL)
    cv2.imshow('diff_homography',diff)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
cv2.destroyAllWindows()
cap.release()
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
#        fgmask = fgbg.apply(frame)
#        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#        img = cv2.add(frame,mask)
#        cv2.namedWindow('background', cv2.WINDOW_NORMAL)
#        cv2.imshow('background',fgmask)
