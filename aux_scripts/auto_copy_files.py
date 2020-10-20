# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:05:27
import argparse
import os
from shutil import copyfile
from tqdm import tqdm as ProgressBar


def copyFiles(input_folder, output_folder, first, last, rate):
    """
    Copy files with a certain rate to a new folder
    """
    copy = False
    counter_files = 0
    counter_copies = 0
    # obtains absolute path of output folder
    output_folder_path = os.path.abspath(output_folder)
    if os.path.isdir(output_folder_path) is not True:
        os.makedirs(output_folder_path)
        input_folder_path = os.path.abspath(input_folder)
        print('folder_path:', input_folder_path)
        file_list = sorted(os.listdir(input_folder_path))
        for filename in ProgressBar(file_list):
            # begins to copy from the file
            if first in filename:
                copy = True
                counter_files += 1
            # copy only
            if copy is True and counter_files % rate is 0:
                # copies file from original folder to the new one
                copyfile(os.path.join(input_folder_path, filename),
                         os.path.join(output_folder_path, filename))
                counter_copies += 1
            # exits when filename has been copied
            elif last in filename:
                copy = False
                print('Files copied:', counter_copies)
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='join files of many folders into a new one')
    parser.add_argument('input_folder', help='input_folder')
    parser.add_argument('output_folder', help='output_folder')
    parser.add_argument('--first', '-f', type=str, help='first file to copy')
    parser.add_argument('--last', '-l', type=str, help='last file to copy')
    parser.add_argument('-r', '--rate', type=int,
                        help='copy one of each RATE files', required=True)
    args = parser.parse_args()
    copyFiles(args.input_folder,
              args.output_folder, args.first, args.last, args.rate)
