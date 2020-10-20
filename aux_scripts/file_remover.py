# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:07:30

import argparse
import os
from tqdm import tqdm as ProgressBar

"""
    This class removes unnecesary files using 3 different modes:
    String mode: if file in folder contains string -> deleted
    Extension: if file has specified extension -> deleted
    Fraction: deleted one of each specified number of files in folder
"""


def removeFiles(args):
    counter_files = 0
    for file in ProgressBar(sorted(os.listdir(args.folder))):
        if args.mode[0] is 'string' or\
           args.mode[0] is 'extension' and args.mode[1] in file:
            os.remove(os.path.abspath(args.folder)+'/'+file)
        elif args.mode[0] == 'fraction':
            fraction = int(args.mode[1])
            if counter_files % fraction == 0:
                os.remove(os.path.abspath(args.folder)+'/'+file)
        elif args.mode[0] == 'range':
            for num_image in range(int(args.mode[1]), int(args.mode[2])+1):
                num_image_str = '{0:06d}'.format(num_image)
                if num_image_str in file:
                    os.remove(os.path.abspath(args.folder)+'/'+file)
        counter_files += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('folder', help='Path to folder')
    parser.add_argument(
            '--mode', type=str, nargs='+',
            help='specifies mode removing: string, range, ext or fraction')
    args = parser.parse_args()
    removeFiles(args)
