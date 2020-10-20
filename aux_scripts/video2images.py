# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:06:07

import argparse
import cv2
import os

"""
Convert the frame for an specified video file to images, with an
determined rate. 
This script is used for dataset generation purposes
"""


if __name__ == "__main__":

    #Command line options's parser
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('video_path', type=str,help='Path to source video')
    parser.add_argument('-r','--rate', type=int,help='Set interval between frames extracted (One of each RATE_NAMER)')
    args = parser.parse_args()
    print("Source Path:", args.video_path)
    
    #Variables
    counter = 0
    counter_images = 0
    
    cap = cv2.VideoCapture(args.video_path) # Video source
    #Image folder creation
    video_name = args.video_path.split('/')[len(args.video_path.split('/'))-1].split('.')[0]
    directory  = os.path.expanduser('~')+'/images_from_video_'+video_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    #Image extraction loop
    print('Images output folder:', directory)
    while True:
        
        r, frame = cap.read()
       
        if r:

            counter += 1

            if(counter%args.rate == 0):
                counter_images += 1
                cv2.imwrite(directory+'/image'+str(counter).rjust(5,'0')+'_'+video_name+'.jpg',frame)
        else:
            break

    print('Images extracted:',counter_images)
