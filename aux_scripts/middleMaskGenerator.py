# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:06:29

import argparse
import cv2
import os
import numpy as np
import zipfile
from tqdm import tqdm as progressBar
from datetime import datetime
"""
Mask generator for reconstruction model from the Japanese (old) change detection method
Zip generator is available in options
"""
#applies a mask of > 255 to the center of each patch
#This generates the input image needed for our Reconstruction Model
def mask2Images(fore_image,back_image):
    #obtains the ROI for an image
    upper_limit = (fore_image.shape[0]*3) // 4
    lower_limit = fore_image.shape[0] // 4
    y_image = np.array(back_image[lower_limit:upper_limit,lower_limit:upper_limit])
    x_image = fore_image
    x_image[lower_limit:upper_limit,lower_limit:upper_limit] = 510 #mask image
    return x_image,y_image

#Applies the mask procedure to an entire dataset
def maskDataset(dataset_folder,create_zip):
    parent_folder = os.path.abspath(os.path.join(dataset_folder, os.pardir))+'/'
    dataset_name  = "_".join(["{:%d_%m-%H}".format(datetime.now()),'dataset'])
    #Creates the zip file of the dataset if options has been specified
    if create_zip != False:
        if os.path.isfile(parent_folder+dataset_name+'.zip'):
            myzip = zipfile.ZipFile(parent_folder+dataset_name+'.zip', 'a',zipfile.ZIP_DEFLATED)
    else:
        myzip = zipfile.ZipFile(parent_folder+dataset_name+'.zip', 'w')
    #Create folder structure needed for the dataset
    if not os.path.isdir(parent_folder+dataset_name):
        os.makedirs(parent_folder+dataset_name+'/input/x/')
        os.makedirs(parent_folder+dataset_name+'/output/y/')
    for file in progressBar(sorted(os.listdir(dataset_folder))):
        #gets the background image from an specific path
        back_file  = file.split('_')[0]+'_background_image.jpg'
        fore_image = cv2.imread(dataset_folder+file)
        back_image = cv2.imread(parent_folder+'patches_background_image/'+back_file)
        x_image,y_image = mask2Images(fore_image,back_image)
        cv2.imwrite(parent_folder+dataset_name+'/input/x/'+file,x_image)
        cv2.imwrite(parent_folder+dataset_name+'/output/y/'+file,y_image)
        #Writes files in zip folder after writting them on dataset folder
        if create_zip != False:
            myzip.write(parent_folder+dataset_name+'/input/x/'+file,dataset_name+'/input/x/'+file)
            myzip.write(parent_folder+dataset_name+'/output/y/'+file,dataset_name+'/output/y/'+file)
    myzip.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate masks for reconstruction')
    parser.add_argument('dataset_folder',help='dataset_folder')
    parser.add_argument('-z','--zip',action='store_true')
    # COMMENTED PART IS FOR BACKGROUND RECONSTRUCTION
    # parser.add_argument('patches_folder',help='Path to patch images')
    # parser.add_argument('background_folder',help='Specify a background patches folder')
    args = parser.parse_args()
    maskDataset(args.dataset_folder,args.zip)