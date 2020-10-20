# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-04-16 17:48:44
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:24:28
# -*- coding: utf-8 -*-
"""
Editor de Spyder

"""

import numpy as np
import cv2
import time

class opticalFlow:
    
    def main(self):
        cap = cv2.VideoCapture('/home/visiona/Vídeos/DJI_0024.MOV')
        counter = 0
#        # params for ShiTomasi corner detection
#        feature_params = dict( maxCorners = 100,
#                               qualityLevel = 0.9,
#                               minDistance = 3,
#                               blockSize = 7 )
#        # Parameters for lucas kanade optical flow
#        lk_params = dict( winSize  = (15,15),
#                          maxLevel = 2,
#                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#        # Create some random colors
#        color = np.random.randint(0,255,(100,3))
#        # Take first frame and find corners in it
        ret, old_frame = cap.read()
#        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#        # Create a mask image for drawing purposes
#        mask = np.zeros_like(old_frame)
        
        cv2.namedWindow("Window",cv2.WINDOW_NORMAL)
        total_start_time = time.time()
        while(ret):
            ret,frame = cap.read()
            counter += 1
            start_time = time.time()
            if(counter%15 == 0):
                resized = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation = cv2.INTER_CUBIC)
                winW,winH = int(resized.shape[1]/12),int(resized.shape[1]/12)
                for (x, y, window) in self.sliding_window(resized, stepSize=int(winW/2), windowSize=(winW, winH)):
            		# if the window does not meet our desired window size, ignore it
                    if (window.shape[0] != winH or window.shape[1] != winW):
                        continue         
    		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    		# WINDOW
     
        		# since we do not have a classifier, we'll just draw the windo
                    clone = resized.copy()
                    rect = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2).copy()
                    
                    cv2.imshow("Window", clone)
                    cv2.waitKey(1)
            print(str(1/(time.time()-start_time))+' fps')
#            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            # calculate optical flow
#            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#            # Select good points
#            good_new = p1[st==1]
#            good_old = p0[st==1]
#            # draw the tracks
#            for i,(new,old) in enumerate(zip(good_new,good_old)):
#                a,b = new.ravel()
#                c,d = old.ravel()
#        #        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#                frame = cv2.circle(frame,(a,b),4,color[i].tolist(),-1)
#            img = cv2.add(frame,mask)
#            cv2.imshow('optical_flow',img)
#            k = cv2.waitKey(30) & 0xff
#            if k == 27:
#                break
#            # Now update the previous frame and previous points
#            old_gray = frame_gray.copy()
#            p0 = good_new.reshape(-1,1,2)
        print('TIEMPO TOTAL (s): '+str(time.time()-total_start_time))
        cv2.destroyAllWindows()
        cap.release()
    
    def sliding_window(self,image, stepSize, windowSize):
    	# slide a window across the image
    	for y in range(0, image.shape[0], stepSize):
    		for x in range(0, image.shape[1], stepSize):
    			# yield the current window
    			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
A = opticalFlow()
A.main()