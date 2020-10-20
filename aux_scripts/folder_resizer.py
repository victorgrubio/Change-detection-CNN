# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:07:23

import cv2
import os
import argparse
from tqdm import tqdm as ProgressBar
"""
Resize the images in an specified folder for a given size.
Rewrite images, it does not create a new folder.s
"""


def resizer(image_path, new_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='resize all images from folder to an specified size'
    )
    parser.add_argument('image_folder',  type=str, help='set root of images')
    parser.add_argument(
            'dst_size', nargs="+", type=int, help='set size of image'
    )
    args = parser.parse_args()
    dst_size = tuple(args.dst_size)
    for image in ProgressBar(sorted(os.listdir(args.image_folder))):
        resizer(args.image_folder+image, dst_size)
