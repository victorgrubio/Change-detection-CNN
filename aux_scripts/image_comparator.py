# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-03 09:34:20

import cv2
import argparse
import sys
from datetime import datetime
from imagePreprocessing import alignImages

"""
Compare two images in order to analyse them for model
testing purposes. It can apply equalization and
"""


# Applies equalization to each color channel of an image
def equalizeImage(image):
    image_eq = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    for channel in range(0, 1):
        image_eq[:, :, channel] = cv2.equalizeHist(image[:, :, channel])
    image_eq = cv2.cvtColor(image_eq, cv2.COLOR_YCrCb2BGR)
    return image_eq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='comparte two images')
    parser.add_argument('image1_path', help='first image path')
    parser.add_argument('image2_path', help='second image path')
    parser.add_argument('--align', action='store_true')
    parser.add_argument('-eq', '--equalize', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', '-s', action='store_true')
    args = parser.parse_args()
    image1 = cv2.imread(args.image1_path)
    image2 = cv2.imread(args.image2_path)
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('imReg2', cv2.WINDOW_NORMAL)
    # Region of each dimension to be cutted in order to
    # delete black regions produced by alignment process
    height_div = image1.shape[0] // 2
    width_div = image1.shape[1] // 4
    if args.align is not False:
        print("Aligning images ...")
        # Registered image will be restored in imReg.
        # The estimated homography will be stored in h.
        imReg2, h = alignImages(image2, image1)
        # Images are cutted in order to delete
        # the black zones produced in alignment
        image1 = image1[
                height_div//4:-(height_div//4),
                width_div//2:-(width_div//2), :
        ]
        imReg2 = imReg2[
                height_div//4:-(height_div//4),
                width_div//2:-(width_div//2), :
        ]
    else:
        # Image is not aligned, so second image is reassing to imReg
        imReg2 = image2
    if args.equalize:
        image1_eq = equalizeImage(image1)
        imReg2_eq = equalizeImage(imReg2)
    else:
        image1_eq = image1
        imReg2_eq = imReg2
    if args.debug:
        counter_patches = 0
        for h in range(0, image1_eq.shape[0], height_div):
            for w in range(0, image1_eq.shape[1], width_div):
                cv2.imshow(
                    'image1', image1_eq[h:h+height_div, w:w+width_div, :]
                )
                cv2.imshow(
                    'imReg2', imReg2_eq[h:h+height_div, w:w+width_div, :]
                )
                k = cv2.waitKey(1)
                if k == 27:
                    sys.exit(0)
                    cv2.destroyAllWindows()
                counter_patches += 1
    date = "{:%H_%M_}".format(datetime.now())
    if args.save:
        cv2.imwrite(date+'fore.jpg', image1_eq)
        cv2.imwrite(date+'back.jpg', imReg2_eq)
    cv2.imshow('image1', image1_eq)
    cv2.imshow('imReg2', imReg2_eq)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
