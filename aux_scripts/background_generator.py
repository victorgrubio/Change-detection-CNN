# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:05:25
import os
import argparse
import cv2
from tqdm import tqdm as ProgressBar


class backgroundGenerator():
    """
    This script creates a copy of a background image for each
    foreground file in a specified folder, for keras training purposes
    """

    def __init__(self, back_image_path, fore_dir):
        self.back_image_path = back_image_path
        self.fore_dir = fore_dir
        self.back_dir = os.path.dirname(back_image_path)

    def main(self):
        # for each fore image, create a new background copy with the same name,
        # in order to have a sorted folder
        # with equal number of files to fore folder
        back_image = cv2.imread(os.path.abspath(self.back_image_path))
        for fore_image in ProgressBar(sorted(os.listdir(self.fore_dir))):
            cv2.imwrite(os.path.abspath(
                    self.back_dir)+'/'+fore_image, back_image
            )
        # removes the original back image to avoid conflicts
        image_filename = self.back_image_path.split('/')[-1]
        if image_filename not in os.listdir(self.fore_dir):
            os.remove(os.path.abspath(self.back_image_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('back_image',
                        help='path of background image to replicate')
    parser.add_argument('fore_folder', help='set path of foreground folder')
    args = parser.parse_args()
    backgroundGen = backgroundGenerator(args.back_image, args.fore_folder)
    backgroundGen.main()
