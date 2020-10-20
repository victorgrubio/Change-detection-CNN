# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:07:34
import argparse
import os
import cv2
import sys
from skimage.measure import compare_ssim
"""
This class obtains the foregroud-background separation from an specified folder
in and interactive way. Obvious splits are made using SSIM.
Non-clear decisions are made manually using keyboard.
(F for fore, B for background) It is used for the old (Japanese Paper) CD Model
"""


# This method stores the images in their corresponding folders
# If not created, generate the folder structure needed
def saveImage(imageA, imageB, folder, filenameA, parent_folder):
    dataset_folder = parent_folder+'/changeDetection_dataset/'
    if not os.path.isdir(dataset_folder):
        os.makedirs(dataset_folder+'input1/back/')
        os.makedirs(dataset_folder+'input1/fore/')
        os.makedirs(dataset_folder+'input2/back/')
        os.makedirs(dataset_folder+'input2/fore/')
    if folder is 'back':
        input1_path = dataset_folder+'input1/back/'+filenameA
        input2_path = dataset_folder+'input2/back/'+filenameA
        cv2.imwrite(input1_path, imageA)
        cv2.imwrite(input2_path, imageB)
    elif folder is 'fore':
        input1_path = dataset_folder+'input1/fore/'+filenameA
        input2_path = dataset_folder+'input2/fore/'+filenameA
        cv2.imwrite(input1_path, imageA)
        cv2.imwrite(input2_path, imageB)


# Read both input images and allows the user to store them using keys b or f
def storeImagesInDataset(imageA_path, imageB_path, mode, parent_folder):
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)
    filenameA = imageA_path.split('/')[len(imageA_path.split('/'))-1]
    score = compareSSIM(imageA, imageB)
    print('score of current images:', score)
    if score > 0.8:
        saveImage(imageA, imageB, 'back', filenameA, parent_folder)
    elif score < 0.3:
        saveImage(imageA, imageB, 'fore', filenameA, parent_folder)
    else:
        cv2.imshow('imageA', imageA)
        cv2.imshow('imageB', imageB)
        key = cv2.waitKey(0)
        if(key is ord('b')):
            saveImage(imageA, imageB, 'back', filenameA, parent_folder)
        elif(key is ord('f')):
            saveImage(imageA, imageB, 'fore', filenameA, parent_folder)
        else:
            cv2.destroyAllWindows()
            sys.exit("no has presionado la tecla correcta")
    print('imagen anidada en el dataset')


# Compute the SSIM between 2 images
# Returns the similarity score
def compareSSIM(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument(
            'patches_folder', type=str, help='Path to patch images'
    )
    parser.add_argument(
            'background_folder', help='Specify a background patches folder'
    )
    parser.add_argument(
            '-m', '--mode', type=str,
            help='set mode: keyboard - ssim', required=True
    )
    args = parser.parse_args()
    # Zip both back and fore folder to compare images easier
    zipped_folders = zip(
            sorted(os.listdir(args.patches_folder)),
            sorted(os.listdir(args.background_folder))
    )
    parent_folder = os.path.abspath(
            os.path.join(args.patches_folder, os.pardir)
    )
    cv2.namedWindow('imageA', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imageA', 660, 660)
    cv2.namedWindow('imageB', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imageB', 660, 660)
    for imageA, imageB in zipped_folders:
        imageA_path = os.path.abspath(args.patches_folder)+'/'+imageA
        imageB_path = os.path.abspath(args.background_folder)+'/'+imageB
        storeImagesInDataset(
                imageA_path, imageB_path, args.mode, parent_folder
        )
