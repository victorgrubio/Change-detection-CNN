# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-05-25 12:44:59
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:24:30
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read video
v_foreground= cv2.VideoCapture("/home/visiona/MEGA/Video/Pruebas Montegancedo/DJI_0001.MOV")
video = cv2.VideoCapture("/home/visiona/MEGA/Video/Pruebas Montegancedo/DJI_0002.MOV")

template1 = cv2.imread('/home/visiona/MEGA/Video/Pruebas Montegancedo/cono_amarillo_1.png',0)
template2 = cv2.imread('/home/visiona/MEGA/Video/Pruebas Montegancedo/cono_amarillo_2.png',0)
template3 = cv2.imread('/home/visiona/MEGA/Video/Pruebas Montegancedo/cono_amarillo_3.png',0)

counter_frames = 0

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('fore',cv2.WINDOW_NORMAL)
cv2.namedWindow('homography',cv2.WINDOW_NORMAL)
cv2.namedWindow('homography1',cv2.WINDOW_NORMAL)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

def IOU(boxA,boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0][0], boxB[0][0])
  yA = max(boxA[0][1], boxB[0][1])
  xB = max(boxA[1][0], boxB[1][0])
  yB = max(boxA[1][1], boxB[1][1])
  # compute the area of intersection rectangle
  interArea = (xB - xA + 1) * (yB - yA + 1)
 
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
  boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
 
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  # return the intersection over union value
  return int(iou*100)


def rectangles2Points(rectangle_array):

  points = []
  for rectangle in rectangle_array:
    for point in rectangle:
      points.append(point)

  points= np.asarray(points)
  return points



def templateMatching(img_rgb,img_gray,template,color):

  w, h = template.shape[::-1]
  # 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
  # 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
  res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

  threshold = 0.63
  counter_templates = 0
  flag_rectangle = 0
  main_rectangles = []
  loc = np.where( res >= threshold)

  for pt in zip(*loc[::-1]):

    my_rectangle = [(pt[0],pt[1]),(pt[0]+w,pt[1]+h)]

    if(flag_rectangle == 0):

      main_rectangles.append(my_rectangle)
      last = len(main_rectangles)-1
      cv2.rectangle(img_rgb, main_rectangles[last][0],main_rectangles[last][1], color, 2)
      flag_rectangle = 1

    else:

      IOU_score =  IOU(my_rectangle,main_rectangles[last])

      if(IOU_score == 0):
        main_rectangles.append(my_rectangle)
        cv2.rectangle(img_rgb, main_rectangles[last][0],main_rectangles[last][1] , color, 2)


  counter_templates = len(main_rectangles)
  #print("NTemplates: ", counter_templates)
  return img_rgb,main_rectangles

while True:
  # Read first frame.
  ok, img_rgb = video.read()
  ok, fore = v_foreground.read()

  if not ok:
    print('Cannot read video file')
    sys.exit()

  img_gray  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
  fore_gray = cv2.cvtColor(fore, cv2.COLOR_BGR2GRAY)

  img_rgb,rectangles1  = templateMatching(img_rgb,img_gray,template1,(0,0,0))
  img_rgb,rectangles2  = templateMatching(img_rgb,img_gray,template2,(255,255,0))

  fore,fore_rectangles1  = templateMatching(fore,fore_gray,template1,(0,0,0))
  fore,fore_rectangles2  = templateMatching(fore,fore_gray,template2,(255,255,0))

  back_rectangles = rectangles1+rectangles2
  fore_rectangles = fore_rectangles1+fore_rectangles2

  print('back_rectangles', len(back_rectangles))
  print('fore_rectangles', len(fore_rectangles))

  back_points = rectangles2Points(back_rectangles)
  fore_points = rectangles2Points(fore_rectangles)


  if(len(back_rectangles) == len(fore_rectangles)):
    h, status = cv2.findHomography(back_points, fore_points)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(img_rgb, h, (fore.shape[1],fore.shape[0]))

    cv2.imshow('homography', im_out)

    h1, statu1s = cv2.findHomography(back_points, fore_points)
     
    # Warp source image to destination based on homography
    im_out1 = cv2.warpPerspective(fore, h1, (img_rgb.shape[1],img_rgb.shape[0]))

    cv2.imshow('homography1', im_out1)

  # img_rgb  = templateMatching(img_rgb,img_gray,template3,(0,0,255))
  counter_frames += 1
  print("frame: ", counter_frames)
  print("--------------------------")


  cv2.imshow('frame',img_rgb)
  cv2.imshow('fore',fore)
  

  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break

cv2.destroyAllWindows()
cap.release()